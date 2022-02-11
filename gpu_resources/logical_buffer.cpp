#include "gpu_resources/logical_buffer.h"

namespace gpu_resources {

LogicalBuffer::LogicalBuffer(vk::DeviceSize size,
                             vk::MemoryPropertyFlags memory_flags)
    : size_(size), memory_flags_(memory_flags) {}

void LogicalBuffer::AddUsage(uint32_t user_ind,
                             vk::BufferUsageFlags usage_flags,
                             vk::AccessFlags2KHR access_flags,
                             vk::PipelineStageFlags2KHR stage_flags) {
  usage_flags_ |= usage_flags;
  access_manager_.AddUsage(
      user_ind,
      ResourceUsage{stage_flags, access_flags, vk::ImageLayout::eUndefined});
}

void LogicalBuffer::Create(std::string debug_name = {}) {
  assert(!buffer_.GetBuffer());
  buffer_ = PhysicalBuffer(size_, usage_flags_, debug_name);
}

void LogicalBuffer::RequestMemory(DeviceMemoryAllocator& allocator) {
  memory_ =
      allocator.RequestMemory(buffer_.GetMemoryRequierments(), memory_flags_);
}

vk::BindBufferMemoryInfo LogicalBuffer::GetBindMemoryInfo() const {
  assert(memory_);
  return buffer_.GetBindMemoryInfo(*memory_);
}

vk::BufferMemoryBarrier2KHR LogicalBuffer::GetPostPassBarrier(
    uint32_t user_ind) {
  auto [src_usage, dst_usage] = access_manager_.GetUserDeps(user_ind);
  if (src_usage.stage == vk::PipelineStageFlagBits2KHR::eNone &&
      dst_usage.stage == vk::PipelineStageFlagBits2KHR::eNone) {
    return {};
  }
  return buffer_.GetBarrier(src_usage.stage, src_usage.access, dst_usage.stage,
                            dst_usage.access);
}

}  // namespace gpu_resources
