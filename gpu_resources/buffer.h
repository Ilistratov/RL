#pragma once

#include <vulkan/vulkan.hpp>

#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class Buffer {
  vk::Buffer buffer_;
  vk::DeviceSize size_;
  MemoryBlock bound_memory_;

 public:
  Buffer() = default;
  Buffer(vk::DeviceSize size, vk::BufferUsageFlags usage_flags);

  Buffer(const Buffer&) = delete;
  void operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept;
  void operator=(Buffer&& other) noexcept;
  void Swap(Buffer& other) noexcept;

  vk::Buffer GetBuffer() const;
  vk::DeviceSize GetSize() const;

  void BindMemory(MemoryBlock memory);
  vk::MemoryRequirements GetMemoryRequierments() const;

  vk::BufferMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags) const;

  ~Buffer();
};

}  // namespace gpu_resources
