#pragma once

#include <vulkan/vulkan.hpp>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class PhysicalBuffer {
  vk::Buffer buffer_ = {};
  vk::DeviceSize size_ = 0;
  vk::BufferUsageFlags usage_flags_ = {};

 public:
  PhysicalBuffer() = default;
  PhysicalBuffer(vk::DeviceSize size,
                 vk::BufferUsageFlags usage_flags,
                 std::string debug_name = {});

  PhysicalBuffer(const PhysicalBuffer&) = delete;
  void operator=(const PhysicalBuffer&) = delete;

  PhysicalBuffer(PhysicalBuffer&& other) noexcept;
  void operator=(PhysicalBuffer&& other) noexcept;
  void Swap(PhysicalBuffer& other) noexcept;

  vk::Buffer GetBuffer() const;
  vk::DeviceSize GetSize() const;
  vk::BufferUsageFlags GetUsageFlags() const;
  vk::MemoryRequirements GetMemoryRequierments() const;
  vk::BindBufferMemoryInfo GetBindMemoryInfo(MemoryBlock memory_block) const;

  vk::BufferMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags) const;

  ~PhysicalBuffer();
};

}  // namespace gpu_resources
