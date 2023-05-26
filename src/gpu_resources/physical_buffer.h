#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gpu_resources {

struct BufferProperties {
  vk::DeviceSize size = 0;
  VmaAllocationCreateFlags allocation_flags = {};
  vk::BufferUsageFlags usage_flags = {};

  static BufferProperties Unite(const BufferProperties& lhs,
                                const BufferProperties& rhs);
};

class PhysicalBuffer {
  BufferProperties properties_ = {};
  uint32_t resource_idx_ = 0;
  vk::Buffer buffer_ = {};
  VmaAllocation allocation_ = {};

  friend class ResourceManager;

  void SetDebugName(const std::string& debug_name) const;

 public:
  PhysicalBuffer() = default;
  PhysicalBuffer(uint32_t resource_idx, BufferProperties properties);

  PhysicalBuffer(const PhysicalBuffer&) = delete;
  void operator=(const PhysicalBuffer&) = delete;

  PhysicalBuffer(PhysicalBuffer&& other) noexcept;
  void operator=(PhysicalBuffer&& other) noexcept;
  void Swap(PhysicalBuffer& other) noexcept;

  ~PhysicalBuffer();

  uint32_t GetIdx() const;
  vk::Buffer GetBuffer() const;
  vk::DeviceSize GetSize() const;
  void* GetMappingStart() const;
  vk::MappedMemoryRange GetMappedMemoryRange() const;

  vk::BufferMemoryBarrier2KHR GenerateBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags) const;
};

}  // namespace gpu_resources
