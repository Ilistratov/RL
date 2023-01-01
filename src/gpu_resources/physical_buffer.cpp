#include "gpu_resources/physical_buffer.h"

#include "base/base.h"
#include "gpu_resources/common.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

BufferProperties BufferProperties::Unite(const BufferProperties& lhs,
                                         const BufferProperties& rhs) {
  return BufferProperties{std::max(lhs.size, rhs.size),
                          lhs.usage_flags | rhs.usage_flags,
                          lhs.memory_flags | rhs.memory_flags};
}

void PhysicalBuffer::CreateVkBuffer() {
  DCHECK(!buffer_) << kErrAlreadyInitialized;
  DCHECK(properties_.size > 0) << kErrCantBeEmpty;
  auto device = base::Base::Get().GetContext().GetDevice();
  buffer_ = device.createBuffer(
      vk::BufferCreateInfo({}, properties_.size, properties_.usage_flags,
                           vk::SharingMode::eExclusive, {}));
}

void PhysicalBuffer::SetDebugName(const std::string& debug_name) const {
  DCHECK(buffer_) << kErrNotInitialized;
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      buffer_.objectType, (uint64_t)(VkBuffer)buffer_, debug_name.c_str()));
}

void PhysicalBuffer::RequestMemory(DeviceMemoryAllocator& allocator) {
  DCHECK(buffer_) << kErrNotInitialized;
  DCHECK(!memory_) << kErrMemoryAlreadyRequested;
  auto device = base::Base::Get().GetContext().GetDevice();
  auto mem_requierments = device.getBufferMemoryRequirements(buffer_);
  memory_ = allocator.RequestMemory(mem_requierments, properties_.memory_flags);
}

vk::BindBufferMemoryInfo PhysicalBuffer::GetBindMemoryInfo() const {
  DCHECK(buffer_) << kErrNotInitialized;
  DCHECK(memory_) << kErrMemoryNotRequested;
  DCHECK(memory_->memory) << kErrMemoryNotAllocated;
  return vk::BindBufferMemoryInfo(buffer_, memory_->memory, memory_->offset);
}

PhysicalBuffer::PhysicalBuffer(uint32_t resource_idx,
                               BufferProperties properties)
    : resource_idx_(resource_idx), properties_(properties) {}

PhysicalBuffer::PhysicalBuffer(PhysicalBuffer&& other) noexcept {
  Swap(other);
}

void PhysicalBuffer::operator=(PhysicalBuffer&& other) noexcept {
  PhysicalBuffer tmp(std::move(other));
  Swap(tmp);
}

void PhysicalBuffer::Swap(PhysicalBuffer& other) noexcept {
  std::swap(resource_idx_, other.resource_idx_);
  std::swap(buffer_, other.buffer_);
  std::swap(properties_, other.properties_);
  std::swap(memory_, other.memory_);
}

PhysicalBuffer::~PhysicalBuffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

uint32_t PhysicalBuffer::GetIdx() const {
  return resource_idx_;
}

vk::Buffer PhysicalBuffer::GetBuffer() const {
  return buffer_;
}

vk::DeviceSize PhysicalBuffer::GetSize() const {
  return properties_.size;
}

void* PhysicalBuffer::GetMappingStart() const {
  DCHECK(memory_) << kErrMemoryNotRequested;
  DCHECK(memory_->memory) << kErrMemoryNotAllocated;
  return memory_->mapping_start;
}

vk::MappedMemoryRange PhysicalBuffer::GetMappedMemoryRange() const {
  DCHECK(memory_) << kErrMemoryNotRequested;
  DCHECK(memory_->memory) << kErrMemoryNotAllocated;
  return vk::MappedMemoryRange(memory_->memory, memory_->offset, memory_->size);
}

vk::BufferMemoryBarrier2KHR PhysicalBuffer::GenerateBarrier(
    vk::PipelineStageFlags2KHR src_stage_flags,
    vk::AccessFlags2KHR src_access_flags,
    vk::PipelineStageFlags2KHR dst_stage_flags,
    vk::AccessFlags2KHR dst_access_flags) const {
  return vk::BufferMemoryBarrier2KHR(src_stage_flags, src_access_flags,
                                     dst_stage_flags, dst_access_flags, {}, {},
                                     GetBuffer(), 0, GetSize());
}

}  // namespace gpu_resources
