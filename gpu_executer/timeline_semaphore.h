#pragma once

#include <vulkan/vulkan.hpp>

namespace gpu_executer {

class TimelineSemaphore {
  vk::Semaphore semaphore_;
  uint64_t counter_ = 0;

 public:
  TimelineSemaphore();

  TimelineSemaphore(const TimelineSemaphore& other) = delete;
  TimelineSemaphore& operator=(TimelineSemaphore& other) = delete;

  vk::Result Wait(uint64_t timeout = UINT64_MAX) const;
  vk::SemaphoreSubmitInfoKHR GetWaitInfo(
      vk::PipelineStageFlags2KHR stage_to_wait_at) const noexcept;
  vk::SemaphoreSubmitInfoKHR GetSignalInfo(
      vk::PipelineStageFlags2KHR stage_to_wait_for);

  ~TimelineSemaphore();
};

}  // namespace gpu_executer
