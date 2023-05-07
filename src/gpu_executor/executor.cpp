#include "gpu_executor/executor.h"

#include <algorithm>
#include <list>

#include "base/base.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace gpu_executor {

bool Executor::TaskInfo::HasSemaphoreOperations() const {
  return external_wait || external_signal;
}

void Executor::ScheduleTask(Task *task, vk::PipelineStageFlags2KHR stage_flags,
                            vk::Semaphore external_signal,
                            vk::Semaphore external_wait,
                            uint32_t secondary_cmd_count) {
  tasks_.push_back({});
  tasks_.back().task = task;
  tasks_.back().stage_flags = stage_flags;
  tasks_.back().external_signal = external_signal;
  tasks_.back().external_wait = external_wait;
  tasks_.back().secondary_cmd_count = secondary_cmd_count;
}

Executor::SubmitInfo Executor::RecordCmdBatch(uint32_t batch_start,
                                              uint32_t batch_end) {
  uint32_t secondary_cmd_count = 0;
  for (uint32_t i = batch_start; i < batch_end; i++) {
    secondary_cmd_count += tasks_[i].secondary_cmd_count;
  }
  vk::CommandBuffer primary_cmd =
      cmd_pool_.GetCmd(vk::CommandBufferLevel::ePrimary, 1)[0];
  SubmitInfo res;
  res.cmd_to_execute = vk::CommandBufferSubmitInfoKHR(primary_cmd);
  res.secondary_cmd =
      cmd_pool_.GetCmd(vk::CommandBufferLevel::eSecondary, secondary_cmd_count);

  primary_cmd.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  uint32_t used_cmd_count = 0;
  for (uint32_t i = batch_start; i < batch_end; i++) {
    uint32_t n_used_cmd_count = used_cmd_count + tasks_[i].secondary_cmd_count;
    std::vector<vk::CommandBuffer> task_secondary_cmd(
        res.secondary_cmd.begin() + used_cmd_count,
        res.secondary_cmd.begin() + n_used_cmd_count);
    used_cmd_count = n_used_cmd_count;

    tasks_[i].task->OnWorkloadRecord(primary_cmd, task_secondary_cmd);
    if (tasks_[i].external_signal) {
      DCHECK(batch_end - batch_start == 1)
          << "Batch with semaphore deps must have only 1 primary cmd";
      res.semaphore_to_signal.push_back(vk::SemaphoreSubmitInfoKHR(
          tasks_[i].external_signal, 0, tasks_[i].stage_flags));
    }
    if (tasks_[i].external_wait) {
      DCHECK(batch_end - batch_start == 1)
          << "Batch with semaphore deps must have only 1 primary cmd";
      res.semaphore_to_wait.push_back(vk::SemaphoreSubmitInfoKHR(
          tasks_[i].external_wait, 0, tasks_[i].stage_flags));
    }
  }
  primary_cmd.end();

  return res;
}

void Executor::Execute() {
  std::list<SubmitInfo> batches;
  std::vector<vk::SubmitInfo2KHR> batch_submit_info;
  size_t primary_cmd_count = 0;
  size_t secondary_cmd_count = 0;
  uint32_t batch_start_ind = 0;

  while (batch_start_ind < tasks_.size()) {
    uint32_t batch_end_ind = batch_start_ind;
    while (batch_end_ind + 1 < tasks_.size() &&
           !tasks_[batch_end_ind + 1].HasSemaphoreOperations()) {
      ++batch_end_ind;
    }
    ++batch_end_ind;

    auto batch = RecordCmdBatch(batch_start_ind, batch_end_ind);

    ++primary_cmd_count;
    secondary_cmd_count += batch.secondary_cmd.size();
    batches.push_back(batch);
    batch_submit_info.push_back(vk::SubmitInfo2KHR(
        {}, batches.back().semaphore_to_wait, batches.back().cmd_to_execute,
        batches.back().semaphore_to_signal));
    batch_start_ind = batch_end_ind;
  }

  auto &context = base::Base::Get().GetContext();
  auto device = context.GetDevice();
  auto fence = device.createFence({});
  context.GetQueue(0).submit2KHR(batch_submit_info, fence);

  std::vector<vk::CommandBuffer> recycle_primary;
  std::vector<vk::CommandBuffer> recycle_secondary;
  recycle_primary.reserve(primary_cmd_count);
  recycle_secondary.reserve(secondary_cmd_count);
  for (auto &batch : batches) {
    recycle_primary.push_back(batch.cmd_to_execute.commandBuffer);
    recycle_secondary.insert(recycle_secondary.end(),
                             batch.secondary_cmd.begin(),
                             batch.secondary_cmd.end());
  }
  cmd_pool_.RecycleCmd(recycle_primary, recycle_secondary, fence);
}

void Executor::ExecuteOneTime(Task *task, uint32_t secondary_cmd_count) {
  vk::CommandBuffer primary_cmd =
      cmd_pool_.GetCmd(vk::CommandBufferLevel::ePrimary, 1)[0];
  std::vector<vk::CommandBuffer> secondary_cmd =
      cmd_pool_.GetCmd(vk::CommandBufferLevel::eSecondary, secondary_cmd_count);

  primary_cmd.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  task->OnWorkloadRecord(primary_cmd, secondary_cmd);
  primary_cmd.end();

  vk::CommandBufferSubmitInfoKHR cmd_submit_info(primary_cmd);
  vk::SubmitInfo2KHR submit_info({}, {}, cmd_submit_info, {});

  auto &context = base::Base::Get().GetContext();
  auto device = context.GetDevice();
  auto fence = device.createFence({});

  context.GetQueue(0).submit2KHR(submit_info, fence);
  auto result = device.waitForFences(fence, true, -1);
  CHECK_VK_RESULT(result) << "Failed to wait for onetime submit";
  device.destroyFence(fence);

  cmd_pool_.RecycleCmd({primary_cmd}, secondary_cmd, {});
}

} // namespace gpu_executor
