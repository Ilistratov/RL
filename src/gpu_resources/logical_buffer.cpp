#include "gpu_resources/logical_buffer.h"

#include "base/base.h"

namespace gpu_resources {

LogicalBuffer::LogicalBuffer(vk::DeviceSize size,
                             vk::MemoryPropertyFlags memory_flags)
    : size_(size), memory_flags_(memory_flags) {}

LogicalBuffer::LogicalBuffer(LogicalBuffer&& other) noexcept {
  Swap(other);
}

void LogicalBuffer::operator=(LogicalBuffer&& other) noexcept {
  LogicalBuffer tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void LogicalBuffer::Swap(LogicalBuffer& other) noexcept {
  buffer_.Swap(other.buffer_);
  std::swap(access_manager_, other.access_manager_);
  std::swap(size_, other.size_);
  std::swap(memory_flags_, other.memory_flags_);
  std::swap(usage_flags_, other.usage_flags_);
  std::swap(memory_, other.memory_);
}

void LogicalBuffer::Create() {
  assert(!buffer_.GetBuffer());
  buffer_ = PhysicalBuffer(size_, usage_flags_);
}

void LogicalBuffer::SetDebugName(const std::string& debug_name) const {
  buffer_.SetDebugName(debug_name);
}

void LogicalBuffer::RequestMemory(DeviceMemoryAllocator& allocator) {
  memory_ =
      allocator.RequestMemory(buffer_.GetMemoryRequierments(), memory_flags_);
}

vk::BindBufferMemoryInfo LogicalBuffer::GetBindMemoryInfo() const {
  assert(memory_);
  return buffer_.GetBindMemoryInfo(*memory_);
}

PhysicalBuffer& LogicalBuffer::GetPhysicalBuffer() {
  return buffer_;
}

void LogicalBuffer::AddUsage(uint32_t user_ind,
                             ResourceUsage usage,
                             vk::BufferUsageFlags buffer_usage_flags) {
  usage.layout = vk::ImageLayout::eUndefined;
  access_manager_.AddUsage(user_ind, usage);
  usage_flags_ |= buffer_usage_flags;
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
