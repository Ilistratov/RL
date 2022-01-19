#include "gpu_resources/buffer.h"

#include "base/base.h"

namespace gpu_resources {

Buffer::Buffer(vk::DeviceSize size, vk::BufferUsageFlags usage_flags)
    : size_(size) {
  assert(size > 0);
  auto device = base::Base::Get().GetContext().GetDevice();
  buffer_ = device.createBuffer(vk::BufferCreateInfo(
      {}, size, usage_flags, vk::SharingMode::eExclusive, {}));
}

Buffer::Buffer(Buffer&& other) noexcept {
  *this = std::move(other);
}

void Buffer::operator=(Buffer&& other) noexcept {
  Buffer tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Buffer::Swap(Buffer& other) noexcept {
  std::swap(buffer_, other.buffer_);
  std::swap(size_, other.size_);
  std::swap(bound_memory_, other.bound_memory_);
}

vk::Buffer Buffer::GetBuffer() const {
  return buffer_;
}

vk::DeviceSize Buffer::GetSize() const {
  return size_;
}

vk::BindBufferMemoryInfo Buffer::GetBindMemoryInfo(MemoryBlock memory) const {
  auto requierments = GetMemoryRequierments();
  assert(memory.size >= requierments.size);
  assert(memory.offset % requierments.alignment == 0);
  return vk::BindBufferMemoryInfo(buffer_, memory.memory, memory.offset);
}

vk::MemoryRequirements Buffer::GetMemoryRequierments() const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getBufferMemoryRequirements(buffer_);
}

vk::BufferMemoryBarrier2KHR Buffer::GetBarrier(
    vk::PipelineStageFlags2KHR src_stage_flags,
    vk::AccessFlags2KHR src_access_flags,
    vk::PipelineStageFlags2KHR dst_stage_flags,
    vk::AccessFlags2KHR dst_access_flags) const {
  return vk::BufferMemoryBarrier2KHR(src_stage_flags, src_access_flags,
                                     dst_stage_flags, dst_access_flags, {}, {},
                                     buffer_, 0, size_);
}

Buffer::~Buffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

}  // namespace gpu_resources
