#include "gpu_resources/buffer_manager.h"

namespace gpu_resources {

BufferUsage& BufferUsage::operator|=(BufferUsage other) {
  access |= other.access;
  stage |= other.stage;
  usage |= other.usage;
  return *this;
}

bool BufferUsage::IsModify() const {
  return (access | (vk::AccessFlagBits2KHR::eAccelerationStructureWrite |
                    vk::AccessFlagBits2KHR::eColorAttachmentWrite |
                    vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite |
                    vk::AccessFlagBits2KHR::eHostWrite |
                    vk::AccessFlagBits2KHR::eMemoryWrite |
                    vk::AccessFlagBits2KHR::eShaderStorageWrite |
                    vk::AccessFlagBits2KHR::eShaderWrite |
                    vk::AccessFlagBits2KHR::eTransferWrite)) !=
         vk::AccessFlagBits2KHR::eNone;
}

bool BufferUsage::IsDependencyNeeded(BufferUsage other) const {
  return IsModify() || other.IsModify();
}

vk::BufferUsageFlags BufferManager::GetAccumulatedUsage() const {
  vk::BufferUsageFlags result;
  for (auto [ind, usage] : usage_by_ind_) {
    result |= usage.usage;
  }
  return result;
}

BufferManager::BufferManager(vk::DeviceSize size,
                             vk::MemoryPropertyFlags memory_properties)
    : size_(size), memory_properties_(memory_properties) {}

void BufferManager::CreateBuffer() {
  assert(!buffer_.GetBuffer());
  buffer_ = Buffer(size_, GetAccumulatedUsage());
}

void BufferManager::ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const {
  allocator.AddMemoryBlock(buffer_.GetMemoryRequierments(), memory_properties_);
}

vk::BindBufferMemoryInfo BufferManager::GetBindMemoryInfo(
    DeviceMemoryAllocator& allocator) const {
  auto memory_block = allocator.GetMemoryBlock(buffer_.GetMemoryRequierments(),
                                               memory_properties_);
  return buffer_.GetBindMemoryInfo(memory_block);
}

std::map<uint32_t, vk::BufferMemoryBarrier2KHR> BufferManager::GetBarriers()
    const {
  std::map<uint32_t, vk::BufferMemoryBarrier2KHR> result;
  for (auto [ind, usage] : usage_by_ind_) {
    auto src_it = usage_by_ind_.find(ind);
    auto dst_it = LoopedNext(src_it);
    if (!usage.IsDependencyNeeded(dst_it->second)) {
      continue;
    }
    auto [src_usage, dst_usage] = GetUsageForBarrier(ind);
    vk::BufferMemoryBarrier2KHR barrier(
        src_usage.stage, src_usage.access, dst_usage.stage, dst_usage.access,
        {}, {}, buffer_.GetBuffer(), 0, buffer_.GetSize());
    result[ind] = barrier;
  }
  return result;
}

Buffer* BufferManager::GetBuffer() {
  return &buffer_;
}

}  // namespace gpu_resources
