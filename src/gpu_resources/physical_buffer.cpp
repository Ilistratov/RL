#include "gpu_resources/physical_buffer.h"

#include "base/base.h"

#include "utill/error_handling.h"

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
  DCHECK(buffer_) << "Resource already created";
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getBufferMemoryRequirements(buffer_);
}

vk::BindBufferMemoryInfo PhysicalBuffer::GetBindMemoryInfo(
    MemoryBlock memory_block) const {
  DCHECK(buffer_) << "Resource must be created to use this method";
  DCHECK(memory_block.memory) << "Expected non-null memory at bind time";
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
  DCHECK(buffer_) << "Resource must be created to use this method";
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      buffer_.objectType, (uint64_t)(VkBuffer)buffer_, debug_name.c_str()));
}

PhysicalBuffer::~PhysicalBuffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

void PhysicalBuffer::RecordCopy(vk::CommandBuffer cmd,
                                const PhysicalBuffer& src,
                                const PhysicalBuffer& dst,
                                vk::DeviceSize src_offset,
                                vk::DeviceSize dst_offset,
                                vk::DeviceSize size) {
  PhysicalBuffer::RecordCopy(cmd, src, dst,
                             {vk::BufferCopy(src_offset, dst_offset, size)});
}

void PhysicalBuffer::RecordCopy(
    vk::CommandBuffer cmd,
    const PhysicalBuffer& src,
    const PhysicalBuffer& dst,
    const std::vector<vk::BufferCopy>& copy_regions) {
  DCHECK(src.buffer_) << "src must be created to use this method";
  DCHECK(dst.buffer_) << "dst must be created to use this method";
  cmd.copyBuffer(src.buffer_, dst.buffer_, copy_regions);
}

}  // namespace gpu_resources
