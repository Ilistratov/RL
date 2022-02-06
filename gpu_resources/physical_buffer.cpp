#include "gpu_resources/physical_buffer.h"

#include "base/base.h"

namespace gpu_resources {

PhysicalBuffer::PhysicalBuffer(vk::DeviceSize size,
                               vk::BufferUsageFlags usage_flags,
                               vk::MemoryPropertyFlags memory_flags)
    : size_(size), usage_flags_(usage_flags), memory_flags_(memory_flags) {
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
  std::swap(bound_memory_, other.bound_memory_);
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

vk::MemoryPropertyFlags PhysicalBuffer::GetMemoryFlags() const {
  return memory_flags_;
}

vk::BindBufferMemoryInfo PhysicalBuffer::GetBindMemoryInfo(
    DeviceMemoryAllocator& device_allocator) const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  auto memory_requierments = device.getBufferMemoryRequirements(buffer_);
  auto memory_block =
      device_allocator.GetMemoryBlock(memory_requierments, memory_flags_);
  return vk::BindBufferMemoryInfo(buffer_, memory_block.memory,
                                  memory_block.offset);
}

void PhysicalBuffer::ReserveMemoryBlock(
    DeviceMemoryAllocator& device_allocator) const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  auto memory_requierments = device.getBufferMemoryRequirements(buffer_);
  device_allocator.AddMemoryBlock(memory_requierments, memory_flags_);
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

PhysicalBuffer::~PhysicalBuffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

}  // namespace gpu_resources
