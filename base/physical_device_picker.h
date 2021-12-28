#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "base/context.h"

namespace base {

class PhysicalDevicePicker {
  const ContextConfig* config_;
  vk::SurfaceKHR surface_;
  std::map<std::string, char> extension_availability_;
  vk::PhysicalDevice result_device_;
  uint32_t result_queue_family_index_ = -1;

  bool CheckFeatures(vk::PhysicalDevice device) const;
  bool CheckPresentModes(vk::PhysicalDevice device) const;
  uint32_t GetSuitableQueueFamilyIndex(vk::PhysicalDevice device) const;
  bool CheckExtensions(vk::PhysicalDevice device);
  bool CheckSurfaceSupport(vk::PhysicalDevice device);
  bool IsDeviceSuitable(vk::PhysicalDevice device);
  uint64_t calcDeviceMemSize(vk::PhysicalDevice device) const;
  bool DeviceCmp(vk::PhysicalDevice lhs, vk::PhysicalDevice rhs) const;
  void PickPhysicalDevice();

 public:
  PhysicalDevicePicker(const ContextConfig* config, vk::SurfaceKHR surface);

  vk::PhysicalDevice GetPickedDevice() const;
  uint32_t GetQueueFamilyIndex() const;
};

}  // namespace base
