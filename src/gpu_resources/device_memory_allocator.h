#pragma once

#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class DeviceMemoryAllocator {
  std::map<uint32_t, MemoryBlock> memory_by_type_ind_;
  vk::PhysicalDeviceMemoryProperties device_memory_properties_;
  std::vector<MemoryBlock> allocations_;

  uint32_t GetSuitableTypeBits(vk::MemoryRequirements requierments,
                               vk::MemoryPropertyFlags property_flags) const;
  uint32_t FindTypeIndex(uint32_t type_bits) const;
  void ExtendPreallocBlock(uint32_t type_index,
                           vk::DeviceSize alignment,
                           vk::DeviceSize size);

 public:
  DeviceMemoryAllocator();

  DeviceMemoryAllocator(const DeviceMemoryAllocator&) = delete;
  void operator=(const DeviceMemoryAllocator&) = delete;

  MemoryBlock* RequestMemory(vk::MemoryRequirements requierments,
                             vk::MemoryPropertyFlags property_flags);
  void Allocate();

  ~DeviceMemoryAllocator();
};

}  // namespace gpu_resources
