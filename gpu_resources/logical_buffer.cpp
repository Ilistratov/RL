#include "gpu_resources/logical_buffer.h"

namespace gpu_resources {

LogicalBuffer::LogicalBuffer(vk::DeviceSize required_size,
                             vk::MemoryPropertyFlags required_memory_flags)
    : required_size_(required_size),
      required_memory_flags_(required_memory_flags) {}

void LogicalBuffer::AddUsage(uint32_t user_ind,
                             vk::BufferUsageFlags usage_flags,
                             vk::AccessFlags2KHR access_flags,
                             vk::PipelineStageFlags2KHR stage_flags) {
  required_usage_flags_ |= usage_flags;
  access_manager_.AddUsage(
      user_ind,
      ResourceUsage{stage_flags, access_flags, vk::ImageLayout::eUndefined});
}

void LogicalBuffer::Create(std::string debug_name = {}) {
  assert(!buffer_.GetBuffer());
  buffer_ = PhysicalBuffer(required_size_, required_usage_flags_, debug_name);
}

void LogicalBuffer::RequestMemory(DeviceMemoryAllocator& allocator) {
  requested_memory_ = allocator.RequestMemory(buffer_.GetMemoryRequierments(),
                                              required_memory_flags_);
}

vk::BindBufferMemoryInfo LogicalBuffer::GetBindMemoryInfo() const {
  assert(requested_memory_);
  return buffer_.GetBindMemoryInfo(*requested_memory_);
}

vk::BufferMemoryBarrier2KHR LogicalBuffer::GetPostPassBarrier(
    uint32_t user_ind) {
  if (dependencies_.empty()) {
    dependencies_ = access_manager_.GetUserDeps();
  }
  assert(next_dep_ind_ < dependencies_.size());
  if (dependencies_[next_dep_ind_].user_id > user_ind) {
    return {};
  }
  assert(dependencies_[next_dep_ind_].user_id == user_ind);

  auto src_usage = dependencies_[next_dep_ind_].src_usage;
  auto dst_usage = dependencies_[next_dep_ind_].dst_usage;
  auto result = buffer_.GetBarrier(src_usage.stage, src_usage.access,
                                   dst_usage.stage, dst_usage.access);
  ++next_dep_ind_;
  if (next_dep_ind_ == dependencies_.size()) {
    next_dep_ind_ = 0;
    // when deps become dynamic, clear |dependencies_| here
  }
  return result;
}

}  // namespace gpu_resources
