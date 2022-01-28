#include "gpu_executer/task.h"

#include <cassert>

#include "base/base.h"
#include "gpu_executer/executer.h"

namespace gpu_executer {

void Task::RecordBarriers() {
  barrier_cmd_.reset();
  vk::CommandBufferInheritanceInfo inheritance_info{};
  barrier_cmd_.begin(vk::CommandBufferBeginInfo({}, &inheritance_info));
  vk::DependencyInfoKHR dep_info(vk::DependencyFlags{}, {}, buffer_barriers_,
                                 image_barriers_);
  barrier_cmd_.pipelineBarrier2KHR(dep_info);
  barrier_cmd_.end();
}

Task::Task(vk::PipelineStageFlags2KHR stage_flags,
           vk::Semaphore external_wait,
           vk::Semaphore external_signal)
    : external_wait_(external_wait),
      external_signal_(external_signal),
      stage_flags_(stage_flags) {}

void Task::AddBarrier(vk::BufferMemoryBarrier2KHR barrier) {
  buffer_barriers_.push_back(barrier);
}

void Task::AddBarrier(vk::ImageMemoryBarrier2KHR barrier) {
  image_barriers_.push_back(barrier);
}

bool Task::IsHasExecutionDep() const {
  return external_wait_ || !executer_->GetTasksToWait(task_ind_).empty();
}

std::vector<vk::SemaphoreSubmitInfoKHR> Task::GetSemaphoresToWait() {
  assert(executer_);
  WaitOnCompletion();
  std::vector<vk::SemaphoreSubmitInfoKHR> result;
  if (external_wait_) {
    result.push_back(
        vk::SemaphoreSubmitInfoKHR(external_wait_, {}, stage_flags_));
  }
  const auto& tasks_to_wait_ind = executer_->GetTasksToWait(task_ind_);
  result.reserve(result.size() + tasks_to_wait_ind.size());
  for (auto wait_task_ind : tasks_to_wait_ind) {
    result.push_back(executer_->GetTaskByInd(wait_task_ind)
                         ->on_complete_semaphore_.GetWaitInfo(stage_flags_));
  }
  return result;
}

std::vector<vk::SemaphoreSubmitInfoKHR> Task::GetSemaphoresToSignal() {
  std::vector<vk::SemaphoreSubmitInfoKHR> result;
  if (external_signal_) {
    result.push_back(
        vk::SemaphoreSubmitInfoKHR(external_signal_, {}, stage_flags_));
  }
  result.push_back(on_complete_semaphore_.GetSignalInfo(stage_flags_));
  return result;
}

void Task::WaitOnCompletion() {
  auto res = on_complete_semaphore_.Wait(5'000'000);
  assert(res == vk::Result::eSuccess);
}

void Task::OnSchedule(Executer* executer, uint32_t task_ind) {
  assert(!executer_);
  assert(executer);
  executer_ = executer;
  task_ind_ = task_ind;
}

void Task::OnCmdCreate(vk::CommandBuffer primary_cmd,
                       vk::CommandBuffer workload_cmd,
                       vk::CommandBuffer barrier_cmd) {
  assert(primary_cmd);
  assert(workload_cmd);
  assert(barrier_cmd);

  primary_cmd_ = primary_cmd;
  workload_cmd_ = workload_cmd;
  barrier_cmd_ = barrier_cmd;
}

void Task::OnRecord() {
  WaitOnCompletion();
  RecordBarriers();
  workload_cmd_.reset();
  vk::CommandBufferInheritanceInfo inheritance_info{};
  workload_cmd_.begin(vk::CommandBufferBeginInfo({}, &inheritance_info));
  OnWorkloadRecord(workload_cmd_);
  workload_cmd_.end();
  primary_cmd_.reset();
  primary_cmd_.begin(vk::CommandBufferBeginInfo{});
  primary_cmd_.executeCommands({workload_cmd_, barrier_cmd_});
  primary_cmd_.end();
}

}  // namespace gpu_executer
