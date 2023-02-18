#include "gpu_resources/physical_buffer.h"

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "base/base.h"
#include "gpu_resources/common.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

BufferProperties BufferProperties::Unite(const BufferProperties& lhs,
                                         const BufferProperties& rhs) {
  return BufferProperties{std::max(lhs.size, rhs.size),
                          lhs.allocation_flags | rhs.allocation_flags,
                          lhs.usage_flags | rhs.usage_flags};
}

void PhysicalBuffer::SetDebugName(const std::string& debug_name) const {
  DCHECK(buffer_) << kErrNotInitialized;
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      buffer_.objectType, (uint64_t)(VkBuffer)buffer_, debug_name.c_str()));
}

PhysicalBuffer::PhysicalBuffer(uint32_t resource_idx,
                               BufferProperties properties)
    : properties_(properties), resource_idx_(resource_idx) {
  VkBufferCreateInfo buffer_create_info =
      vk::BufferCreateInfo{{}, properties_.size, properties_.usage_flags};
  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.flags = properties_.allocation_flags;
  allocation_create_info.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO;
  VmaAllocator allocator = base::Base::Get().GetContext().GetAllocator();
  VkBuffer res_buffer;
  auto create_res =
      vmaCreateBuffer(allocator, &buffer_create_info, &allocation_create_info,
                      &res_buffer, &allocation_, nullptr);
  CHECK(create_res == VK_SUCCESS)
      << "Failed to create buffer: " << vk::to_string((vk::Result)create_res);
  buffer_ = res_buffer;
}

PhysicalBuffer::PhysicalBuffer(PhysicalBuffer&& other) noexcept {
  Swap(other);
}

void PhysicalBuffer::operator=(PhysicalBuffer&& other) noexcept {
  PhysicalBuffer tmp(std::move(other));
  Swap(tmp);
}

void PhysicalBuffer::Swap(PhysicalBuffer& other) noexcept {
  std::swap(properties_, other.properties_);
  std::swap(resource_idx_, other.resource_idx_);
  std::swap(buffer_, other.buffer_);
  std::swap(allocation_, other.allocation_);
}

PhysicalBuffer::~PhysicalBuffer() {
  VmaAllocator allocator = base::Base::Get().GetContext().GetAllocator();
  vmaDestroyBuffer(allocator, buffer_, allocation_);
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
  DCHECK(buffer_) << kErrNotInitialized;
  DCHECK(allocation_) << kErrMemoryNotAllocated;
  VmaAllocator allocator = base::Base::Get().GetContext().GetAllocator();
  VmaAllocationInfo alloc_info{};
  vmaGetAllocationInfo(allocator, allocation_, &alloc_info);
  return alloc_info.pMappedData;
}

vk::MappedMemoryRange PhysicalBuffer::GetMappedMemoryRange() const {
  DCHECK(buffer_) << kErrNotInitialized;
  DCHECK(allocation_) << kErrMemoryNotAllocated;
  VmaAllocator allocator = base::Base::Get().GetContext().GetAllocator();
  VmaAllocationInfo alloc_info{};
  vmaGetAllocationInfo(allocator, allocation_, &alloc_info);
  return vk::MappedMemoryRange(alloc_info.deviceMemory, alloc_info.offset,
                               alloc_info.size);
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
