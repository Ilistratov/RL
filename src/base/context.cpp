#include "base/context.h"
#include <vulkan/vulkan_handles.hpp>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

#include "base/base.h"
#include "base/physical_device_picker.h"
#include "utill/error_handling.h"
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
  vk::PhysicalDevice8BitStorageFeaturesKHR storage_features(true, true, true);

  vk::StructureChain info_chain(device_info, timeline_semaphore_features,
                                synchronization2_features, storage_features);

  LOG << "Creating device with:\nExt: " << config.device_extensions
      << "\nLayers: " << config.device_layers;

  device_ = physical_device_.createDevice(info_chain.get());

  device_queues_.resize(config.queue_count);
  for (uint32_t q_ind = 0; q_ind < device_queues_.size(); q_ind++) {
    device_queues_[q_ind] = device_.getQueue(queue_family_index_, q_ind);
    CHECK(device_queues_[q_ind])
        << "failed to retrive " << q_ind << "'th device queue";
  }
}

void Context::InitializeAllocator() {
  VmaVulkanFunctions vulkanFunctions = {};
  vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocateCreateInfo{
      .physicalDevice = physical_device_,
      .device = device_,
      .pVulkanFunctions = &vulkanFunctions,
      .instance = base::Base::Get().GetInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_2};
  auto result = vmaCreateAllocator(&allocateCreateInfo, &allocator_);
  CHECK(result == VK_SUCCESS) << "Failed to create vma allocator: "
                              << vk::to_string((vk::Result)result);
}

Context::Context(ContextConfig config) {
  LOG << "Picking physical device";
  PickPhysicalDevice(config);
  LOG << "Creating device";
  CreateDevice(config);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);
  LOG << "Initializing vma";
  InitializeAllocator();
  LOG << "Device context initialized";
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
  std::swap(allocator_, other.allocator_);
  std::swap(queue_family_index_, other.queue_family_index_);
  device_queues_.swap(other.device_queues_);
}

vk::PhysicalDevice Context::GetPhysicalDevice() const {
  return physical_device_;
}

vk::Device Context::GetDevice() const {
  return device_;
}

VmaAllocator Context::GetAllocator() const {
  return allocator_;
}

uint32_t Context::GetQueueFamilyIndex() const {
  return queue_family_index_;
}

vk::Queue Context::GetQueue(uint32_t queue_ind) const {
  DCHECK(queue_ind < device_queues_.size())
      << "queue_ind " << queue_ind << "out of range.";
  return device_queues_[queue_ind];
}

Context::~Context() {
  if (allocator_) {
    vmaDestroyAllocator(allocator_);
  }
  if (device_) {
    device_.destroy();
  }
}

}  // namespace base
