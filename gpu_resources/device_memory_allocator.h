#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class DeviceMemoryAllocator {
  std::map<uint32_t, MemoryBlock> memory_by_type_ind_;
  vk::PhysicalDeviceMemoryProperties device_memory_properties_;

  bool IsMemoryTypeSuitable(uint32_t type_index,
                            uint32_t type_bits,
                            vk::MemoryPropertyFlags property_flags) const;

 public:
  DeviceMemoryAllocator();

  DeviceMemoryAllocator(const DeviceMemoryAllocator&) = delete;
  void operator=(const DeviceMemoryAllocator&) = delete;

  uint32_t FindMemoryTypeIndex(vk::MemoryRequirements requierments,
                               vk::MemoryPropertyFlags property_flags) const;

  void AddMemoryBlock(vk::MemoryRequirements requierments,
                      vk::MemoryPropertyFlags property_flags);
  void Allocate();
  MemoryBlock GetMemoryBlock(vk::MemoryRequirements requierments,
                             vk::MemoryPropertyFlags property_flags);

  ~DeviceMemoryAllocator();
};

}  // namespace gpu_resources
