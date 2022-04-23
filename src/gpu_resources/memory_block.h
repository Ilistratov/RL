#pragma once

#include <vulkan/vulkan.hpp>

namespace gpu_resources {

struct MemoryBlock {
  vk::DeviceMemory memory = {};
  vk::DeviceSize size = 0;
  vk::DeviceSize offset = 0;
  uint32_t type_index = UINT32_MAX;
  void* mapping_start = nullptr;

  vk::DeviceSize GetAlignedOffset(vk::DeviceSize alignment) const;
  MemoryBlock Suballocate(vk::DeviceSize block_size, vk::DeviceSize alignment);
};

}  // namespace gpu_resources
