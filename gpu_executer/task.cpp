#include "gpu_executer/task.h"

#include <cassert>

#include "base/base.h"
#include "gpu_executer/executer.h"

namespace gpu_executer {

void Task::RecordSrcDep(uint32_t dst_task_ind, const BarrierDep& dep) const {
  auto buffer_barriers = executer_->GetBufferBarriers(dep.buffer_barrier_ind);
  auto image_barriers = executer_->GetImageBarriers(dep.image_barrier_ind);
  vk::DependencyInfoKHR dep_info({}, {}, buffer_barriers, image_barriers);
  auto event = src_events_by_task_ind_.at(dst_task_ind);
  src_dep_cmd_.setEvent2KHR(event, dep_info);
};

void Task::RecordDstDep(uint32_t src_task_ind, const BarrierDep& dep) const {
  auto buffer_barriers = executer_->GetBufferBarriers(dep.buffer_barrier_ind);
  auto image_barriers = executer_->GetImageBarriers(dep.image_barrier_ind);
  vk::DependencyInfoKHR dep_info({}, {}, buffer_barriers, image_barriers);
  auto event = executer_->GetTaskByInd(src_task_ind)
                   ->src_events_by_task_ind_.at(task_ind_);
  dst_dep_cmd_.waitEvents2KHR(event, dep_info);
}

Task::Task(vk::PipelineStageFlags2KHR stage_flags,
           vk::Semaphore external_wait,
           vk::Semaphore external_signal)
    : stage_flags_(stage_flags),
      external_wait_(external_wait),
      external_signal_(external_signal) {}

Task::BarrierDep& Task::GetSrcDep(uint32_t dst_task_ind) {
  return src_deps_by_task_ind_[dst_task_ind];
}

Task::BarrierDep& Task::GetDstDep(uint32_t src_task_ind) {
  return dst_deps_by_task_ind_[src_task_ind];
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
  on_complete_semaphore_.Wait();
}

void Task::OnSchedule(Executer* executer, uint32_t task_ind) {
  assert(!executer_);
  assert(executer);
  executer_ = executer;
  task_ind_ = task_ind;
}

void Task::OnCmdCreate(vk::CommandBuffer primary_cmd,
                       std::array<vk::CommandBuffer, 3> secondary_cmd) {
  for (const auto& cb : secondary_cmd) {
    assert(cb);
  }
  assert(primary_cmd);

  src_dep_cmd_ = secondary_cmd[0];
  workload_cmd_ = secondary_cmd[1];
  dst_dep_cmd_ = secondary_cmd[2];
  primary_cmd_ = primary_cmd;

  auto device = base::Base::Get().GetContext().GetDevice();
  for (const auto& [dst_task_ind, dep] : src_deps_by_task_ind_) {
    if (!src_events_by_task_ind_.contains(dst_task_ind)) {
      src_events_by_task_ind_[dst_task_ind] = device.createEvent(
          vk::EventCreateInfo(vk::EventCreateFlagBits::eDeviceOnlyKHR));
    }
  }
}

void Task::OnRecord() {
  dst_dep_cmd_.begin(vk::CommandBufferBeginInfo{});
  for (const auto& [src_task_ind, dep] : dst_deps_by_task_ind_) {
    RecordDstDep(src_task_ind, dep);
  }
  dst_dep_cmd_.end();

  src_dep_cmd_.begin(vk::CommandBufferBeginInfo{});
  for (const auto& [dst_task_ind, dep] : src_deps_by_task_ind_) {
    RecordSrcDep(dst_task_ind, dep);
  }
  src_dep_cmd_.end();

  OnWorkloadRecord(workload_cmd_);

  primary_cmd_.begin(vk::CommandBufferBeginInfo{});
  primary_cmd_.executeCommands({dst_dep_cmd_, workload_cmd_, src_dep_cmd_});
  primary_cmd_.end();
}

Task::~Task() {
  auto device = base::Base::Get().GetContext().GetDevice();
  for (auto [ind, event] : src_events_by_task_ind_) {
    device.destroyEvent(event);
  }
}

}  // namespace gpu_executer
