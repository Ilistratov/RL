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

void Buffer::BindMemory(MemoryBlock memory) {
  assert(!bound_memory_.memory);
  assert(memory.size >= size_);
  bound_memory_ = memory;
  auto device = base::Base::Get().GetContext().GetDevice();
  device.bindBufferMemory(buffer_, bound_memory_.memory, bound_memory_.offset);
}

vk::MemoryRequirements Buffer::GetMemoryRequierments() const {
  assert(buffer_);
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getBufferMemoryRequirements(buffer_);
}

Buffer::~Buffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

}  // namespace gpu_resources
