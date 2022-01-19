#include "gpu_executer/executer.h"

#include <algorithm>

#include "base/base.h"
#include "utill/logger.h"

namespace gpu_executer {

void Executer::CreateCommandBuffers() {
  LOG(INFO) << "Creating gpu Executer command buffers";
  assert(task_primary_cmd_.empty());
  auto& context = base::Base::Get().GetContext();
  cmd_pool_ = context.GetDevice().createCommandPool(
      vk::CommandPoolCreateInfo({}, context.GetQueueFamilyIndex()));
  task_primary_cmd_ =
      context.GetDevice().allocateCommandBuffers(vk::CommandBufferAllocateInfo(
          cmd_pool_, vk::CommandBufferLevel::ePrimary, tasks_.size()));
  task_secondary_cmd_ =
      context.GetDevice().allocateCommandBuffers(vk::CommandBufferAllocateInfo(
          cmd_pool_, vk::CommandBufferLevel::eSecondary, 2 * tasks_.size()));

  for (uint32_t task_id = 0; task_id < tasks_.size(); task_id++) {
    tasks_[task_id].task->OnCmdCreate(task_primary_cmd_[task_id],
                                      task_secondary_cmd_[task_id * 2 + 0],
                                      task_secondary_cmd_[task_id * 2 + 1]);
  }
}

void Executer::CleanupTaskWaitInd() {
  LOG(INFO) << "Cleaning up gpu Executer tasks wait ind";
  for (auto& task : tasks_) {
    std::sort(task.wait_task_ind.begin(), task.wait_task_ind.end());
    auto it = std::unique(task.wait_task_ind.begin(), task.wait_task_ind.end());
    task.wait_task_ind.resize(it - task.wait_task_ind.begin());
    task.wait_task_ind.shrink_to_fit();
  }
}

Executer::SubmitInfo Executer::GetBatchSubmitInfo(uint32_t batch_start,
                                                  uint32_t batch_end) {
  SubmitInfo result;
  for (uint32_t task_ind = batch_start; task_ind < batch_end; ++task_ind) {
    auto& task = tasks_[task_ind];
    task.task->OnRecord();
    result.cmd_to_execute.push_back(task_primary_cmd_[task_ind]);

    auto semaphores_to_wait = task.task->GetSemaphoresToWait();
    assert(semaphores_to_wait.empty() || result.semaphores_to_wait.empty());
    result.semaphores_to_wait = semaphores_to_wait;

    auto semaphores_to_signal = task.task->GetSemaphoresToSignal();
    for (auto semaphore_submit_info : semaphores_to_signal) {
      result.semaphores_to_signal.push_back(semaphore_submit_info);
    }
  }
  return result;
}

Task* Executer::GetTaskByInd(uint32_t id) const {
  return tasks_.at(id).task.get();
}

const std::vector<uint32_t>& Executer::GetTasksToWait(
    uint32_t dst_task_ind) const {
  assert(dst_task_ind < tasks_.size());
  return tasks_[dst_task_ind].wait_task_ind;
}

void Executer::AddBarrier(uint32_t src_task_ind,
                          vk::BufferMemoryBarrier2KHR barrier) {
  assert(src_task_ind < tasks_.size());
  tasks_[src_task_ind].task->AddBarrier(barrier);
}
void Executer::AddBarrier(uint32_t src_task_ind,
                          vk::ImageMemoryBarrier2KHR barrier) {
  assert(src_task_ind < tasks_.size());
  tasks_[src_task_ind].task->AddBarrier(barrier);
}

void Executer::AddExecutionDependency(uint32_t src_task_ind,
                                      uint32_t dst_task_ind) {
  tasks_[dst_task_ind].wait_task_ind.push_back(src_task_ind);
}

uint32_t Executer::ScheduleTask(std::unique_ptr<Task> task) {
  uint32_t task_ind = tasks_.size();
  tasks_.push_back({});
  auto& task_info = tasks_[task_ind];
  task_info.task = std::move(task);
  task_info.task->OnSchedule(this, task_ind);
  return task_ind;
}

void Executer::PreExecuteInit() {
  LOG(INFO) << "Initializing gpu Executer";
  assert(!tasks_.empty());
  CreateCommandBuffers();
  CleanupTaskWaitInd();
  LOG(INFO) << "Initialized gpu Executer";
}

void Executer::Execute() {
  std::vector<Executer::SubmitInfo> batch;
  std::vector<vk::SubmitInfo2KHR> batch_submit_info;
  uint32_t batch_start_ind = 0;

  while (batch_start_ind < tasks_.size()) {
    uint32_t batch_end_ind = batch_start_ind;
    while (batch_end_ind + 1 < tasks_.size() &&
           !tasks_[batch_end_ind].task->IsHasExecutionDep()) {
      ++batch_end_ind;
    }
    ++batch_end_ind;

    batch.push_back(GetBatchSubmitInfo(batch_start_ind, batch_end_ind));
    batch_start_ind = batch_end_ind;
    batch_submit_info.push_back(vk::SubmitInfo2KHR(
        {}, batch.back().semaphores_to_wait, batch.back().cmd_to_execute,
        batch.back().semaphores_to_signal));
  }

  auto& context = base::Base::Get().GetContext();
  context.GetQueue(0).submit2KHR(batch_submit_info);
}

Executer::~Executer() {
  for (auto& task : tasks_) {
    task.task->WaitOnCompletion();
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.freeCommandBuffers(cmd_pool_, task_primary_cmd_);
  device.freeCommandBuffers(cmd_pool_, task_secondary_cmd_);
  device.destroyCommandPool(cmd_pool_);
}

}  // namespace gpu_executer
