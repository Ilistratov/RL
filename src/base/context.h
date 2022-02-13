#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW\glfw3.h>

namespace base {

struct ContextConfig {
  std::vector<const char*> device_layers;
  std::vector<const char*> device_extensions;
  // I'm lazy so i'll create 'queue_count' queues from one queue family that has
  // all flags from 'required_flags'
  uint32_t queue_count;
  vk::QueueFlags required_flags;
};

/*
 * 'Context' class is responsible for creating and managing 'vk::Device',
 * and device queues
 */
class Context {
  vk::Device device_;
  vk::PhysicalDevice physical_device_;
  uint32_t queue_family_index_ = -1;
  std::vector<vk::Queue> device_queues_;

  void PickPhysicalDevice(ContextConfig& config);
  void CreateDevice(ContextConfig& config);

 public:
  Context() = default;
  Context(ContextConfig config);

  Context(Context&& other) noexcept;
  void operator=(Context&& other) noexcept;
  void Swap(Context& other) noexcept;

  vk::PhysicalDevice GetPhysicalDevice() const;
  vk::Device GetDevice() const;
  uint32_t GetQueueFamilyIndex() const;
  vk::Queue GetQueue(uint32_t queue_ind) const;

  ~Context();
};

}  // namespace base
