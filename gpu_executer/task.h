#pragma once

#include <array>
#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_executer/timeline_semaphore.h"

namespace gpu_executer {

class Executer;

class Task {
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers_;
  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers_;

  vk::CommandBuffer primary_cmd_;
  vk::CommandBuffer workload_cmd_;
  vk::CommandBuffer barrier_cmd_;

  TimelineSemaphore on_complete_semaphore_;
  vk::Semaphore external_wait_;
  vk::Semaphore external_signal_;
  vk::PipelineStageFlags2KHR stage_flags_;

  Executer* executer_ = nullptr;
  uint32_t task_ind_ = UINT32_MAX;

  void RecordBarriers();

 public:
  Task(vk::PipelineStageFlags2KHR stage_flags,
       vk::Semaphore external_wait = {},
       vk::Semaphore external_signal = {});

  void AddBarrier(vk::BufferMemoryBarrier2KHR barrier);
  void AddBarrier(vk::ImageMemoryBarrier2KHR barrier);

  bool IsHasExecutionDep() const;
  std::vector<vk::SemaphoreSubmitInfoKHR> GetSemaphoresToWait();
  std::vector<vk::SemaphoreSubmitInfoKHR> GetSemaphoresToSignal();
  void WaitOnCompletion();

  void OnSchedule(Executer* executer, uint32_t task_ind);
  void OnCmdCreate(vk::CommandBuffer primary_cmd,
                   vk::CommandBuffer workload_cmd,
                   vk::CommandBuffer barrier_cmd);
  void OnRecord();
  virtual void OnWorkloadRecord(vk::CommandBuffer cmd) = 0;

  virtual ~Task() = default;
};

template <typename T>
concept CmdInvocable = std::is_invocable<T, vk::CommandBuffer>::value;

template <CmdInvocable Func>
class LambdaTask : public Task {
  Func func_;

 public:
  LambdaTask(vk::PipelineStageFlags2KHR stage_flags,
             Func func,
             vk::Semaphore external_wait = {},
             vk::Semaphore external_signal = {})
      : Task(stage_flags, external_wait, external_signal), func_(func) {}

  void OnWorkloadRecord(vk::CommandBuffer cmd) override { func_(cmd); }
};

}  // namespace gpu_executer
