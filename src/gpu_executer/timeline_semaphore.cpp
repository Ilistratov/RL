#include "gpu_executer/timeline_semaphore.h"

#include "base/base.h"

namespace gpu_executer {

TimelineSemaphore::TimelineSemaphore() {
  vk::SemaphoreTypeCreateInfo timeline_create_info(vk::SemaphoreType::eTimeline,
                                                   counter_);
  vk::SemaphoreCreateInfo semaphore_create_info{};
  semaphore_create_info.pNext = &timeline_create_info;
  auto device = base::Base::Get().GetContext().GetDevice();
  semaphore_ = device.createSemaphore(semaphore_create_info);
}

vk::Result TimelineSemaphore::Wait(uint64_t timeout) const {
  assert(semaphore_);
  vk::SemaphoreWaitInfo wait_info({}, semaphore_, counter_);
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.waitSemaphores(wait_info, timeout);
}

vk::SemaphoreSubmitInfoKHR TimelineSemaphore::GetWaitInfo(
    vk::PipelineStageFlags2KHR stage_to_wait_at) const noexcept {
  return vk::SemaphoreSubmitInfoKHR(semaphore_, counter_, stage_to_wait_at);
}

vk::SemaphoreSubmitInfoKHR TimelineSemaphore::GetSignalInfo(
    vk::PipelineStageFlags2KHR stage_to_wait_for) {
  Wait();
  ++counter_;
  return vk::SemaphoreSubmitInfoKHR(semaphore_, counter_, stage_to_wait_for);
}

TimelineSemaphore::~TimelineSemaphore() {
  Wait();
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroySemaphore(semaphore_);
}

}  // namespace gpu_executer
