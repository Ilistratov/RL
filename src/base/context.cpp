#include "base/context.h"

#include "base/base.h"
#include "base/physical_device_picker.h"
#include "utill/logger.h"

namespace base {

void Context::PickPhysicalDevice(ContextConfig& config) {
  PhysicalDevicePicker picker(&config, Base::Get().GetWindow().GetSurface());
  physical_device_ = picker.GetPickedDevice();
  queue_family_index_ = picker.GetQueueFamilyIndex();
}

void Context::CreateDevice(ContextConfig& config) {
  std::vector<float> queue_priorities(config.queue_count, 1.0);
  vk::DeviceQueueCreateInfo queue_create_info(
      vk::DeviceQueueCreateFlags{}, queue_family_index_, queue_priorities);
  vk::PhysicalDeviceFeatures device_features;
  device_features.geometryShader = true;
  vk::DeviceCreateInfo device_info(vk::DeviceCreateFlags{}, queue_create_info,
                                   config.device_layers,
                                   config.device_extensions, &device_features);
  vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features(true);
  vk::PhysicalDeviceSynchronization2FeaturesKHR synchronization2_features(true);

  vk::StructureChain<vk::DeviceCreateInfo,
                     vk::PhysicalDeviceTimelineSemaphoreFeatures,
                     vk::PhysicalDeviceSynchronization2FeaturesKHR>
      info_chain(device_info, timeline_semaphore_features,
                 synchronization2_features);

  LOG << "Creating device with:\nExt: " << config.device_extensions
      << "\nLayers: " << config.device_layers;

  device_ = physical_device_.createDevice(info_chain.get());
  assert(device_);

  device_queues_.resize(config.queue_count);
  for (uint32_t q_ind = 0; q_ind < device_queues_.size(); q_ind++) {
    device_queues_[q_ind] = device_.getQueue(queue_family_index_, q_ind);
    assert(device_queues_[q_ind]);
  }
}

Context::Context(ContextConfig config) {
  LOG << "Picking physical device";
  PickPhysicalDevice(config);
  LOG << "Creating device";
  CreateDevice(config);
  LOG << "Initialized context";
}

Context::Context(Context&& other) noexcept {
  *this = std::move(other);
}

void Context::operator=(Context&& other) noexcept {
  Context tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Context::Swap(Context& other) noexcept {
  std::swap(device_, other.device_);
  std::swap(physical_device_, other.physical_device_);
  std::swap(queue_family_index_, other.queue_family_index_);
  device_queues_.swap(other.device_queues_);
}

vk::PhysicalDevice Context::GetPhysicalDevice() const {
  return physical_device_;
}

vk::Device Context::GetDevice() const {
  return device_;
}

uint32_t Context::GetQueueFamilyIndex() const {
  return queue_family_index_;
}

vk::Queue Context::GetQueue(uint32_t queue_ind) const {
  assert(queue_ind < device_queues_.size());
  return device_queues_[queue_ind];
}

Context::~Context() {
  if (!device_) {
    return;
  }
  device_.destroy();
}

}  // namespace base
