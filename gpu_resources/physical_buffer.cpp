#include "gpu_resources/physical_buffer.h"

#include "base/base.h"

namespace gpu_resources {

PhysicalBuffer::PhysicalBuffer(vk::DeviceSize size,
                               vk::BufferUsageFlags usage_flags)
    : size_(size), usage_flags_(usage_flags) {
  assert(size > 0);
  auto device = base::Base::Get().GetContext().GetDevice();
  buffer_ = device.createBuffer(vk::BufferCreateInfo(
      {}, size, usage_flags, vk::SharingMode::eExclusive, {}));
}

PhysicalBuffer::PhysicalBuffer(PhysicalBuffer&& other) noexcept {
  Swap(other);
}

void PhysicalBuffer::operator=(PhysicalBuffer&& other) noexcept {
  PhysicalBuffer tmp(std::move(other));
  Swap(tmp);
}

void PhysicalBuffer::Swap(PhysicalBuffer& other) noexcept {
  std::swap(buffer_, other.buffer_);
  std::swap(size_, other.size_);
  std::swap(usage_flags_, other.usage_flags_);
}

vk::Buffer PhysicalBuffer::GetBuffer() const {
  return buffer_;
}

vk::DeviceSize PhysicalBuffer::GetSize() const {
  return size_;
}

vk::BufferUsageFlags PhysicalBuffer::GetUsageFlags() const {
  return usage_flags_;
}

vk::MemoryRequirements PhysicalBuffer::GetMemoryRequierments() const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getBufferMemoryRequirements(buffer_);
}

vk::BindBufferMemoryInfo PhysicalBuffer::GetBindMemoryInfo(
    MemoryBlock memory_block) const {
  assert(buffer_);
  assert(memory_block.memory);
  return vk::BindBufferMemoryInfo(buffer_, memory_block.memory,
                                  memory_block.offset);
}

vk::BufferMemoryBarrier2KHR PhysicalBuffer::GetBarrier(
    vk::PipelineStageFlags2KHR src_stage_flags,
    vk::AccessFlags2KHR src_access_flags,
    vk::PipelineStageFlags2KHR dst_stage_flags,
    vk::AccessFlags2KHR dst_access_flags) const {
  return vk::BufferMemoryBarrier2KHR(src_stage_flags, src_access_flags,
                                     dst_stage_flags, dst_access_flags, {}, {},
                                     buffer_, 0, size_);
}

void PhysicalBuffer::SetDebugName(const std::string& debug_name) const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      buffer_.objectType, (uint64_t)(VkBuffer)buffer_, debug_name.c_str()));
}

PhysicalBuffer::~PhysicalBuffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

}  // namespace gpu_resources
