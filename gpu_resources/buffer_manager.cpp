#include "gpu_resources/buffer_manager.h"

namespace gpu_resources {

BufferManager::Usage& BufferManager::Usage::operator|=(
    const BufferManager::Usage& other) {
  access |= other.access;
  stage |= other.stage;
  usage |= other.usage;
  return *this;
}

bool BufferManager::Usage::IsModify() const {
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

vk::BufferUsageFlags BufferManager::GetAccumulatedUsage() const {
  vk::BufferUsageFlags result;
  for (auto [ind, usage] : usage_by_ind_) {
    result |= usage.usage;
  }
  return result;
}

BufferManager::Usage BufferManager::GetDstUsage(uint32_t src_user_ind) const {
  auto it = usage_by_ind_.find(src_user_ind);
  assert(it != usage_by_ind_.end());
  ++it;
  Usage result;
  if (it != usage_by_ind_.end()) {
    result = it->second;
  }
  while (it != usage_by_ind_.end() && !it->second.IsModify()) {
    result |= it->second;
    ++it;
  }
  return result;
}

BufferManager::Usage BufferManager::GetSrcUsage(uint32_t src_user_ind) const {
  auto it = usage_by_ind_.find(src_user_ind);
  assert(it != usage_by_ind_.end());
  Usage result;
  if (it == usage_by_ind_.begin()) {
    return result;
  }
  --it;
  result = it->second;
  while (it != usage_by_ind_.begin()) {
    --it;
    if (it->second.IsModify()) {
      break;
    }
    result |= it->second;
  }
  return result;
}

BufferManager::BufferManager(vk::DeviceSize size,
                             vk::MemoryPropertyFlags memory_properties)
    : size_(size), memory_properties_(memory_properties) {}

void BufferManager::AddUsage(uint32_t user_ind, BufferManager::Usage usage) {
  assert(!usage_by_ind_.contains(user_ind));
  usage_by_ind_[user_ind] = usage;
}

void BufferManager::CreateBuffer() {
  buffer_ = Buffer(size_, GetAccumulatedUsage());
}

void BufferManager::ReserveMemoryBlock(DeviceMemoryAllocator& allocator) {
  allocator.AddMemoryBlock(buffer_.GetMemoryRequierments(), memory_properties_);
}

void BufferManager::AllocateAndBindMemory(DeviceMemoryAllocator& allocator) {
  auto memory_block = allocator.GetMemoryBlock(buffer_.GetMemoryRequierments(),
                                               memory_properties_);
  buffer_.BindMemory(memory_block);
}

std::map<uint32_t, vk::BufferMemoryBarrier2KHR> BufferManager::GetBarriers()
    const {
  std::map<uint32_t, vk::BufferMemoryBarrier2KHR> result;
  for (auto [ind, usage] : usage_by_ind_) {
    if (usage.IsModify()) {
      auto dst_usage = GetDstUsage(ind);
      result[ind] = buffer_.GetBarrier(usage.stage, usage.access,
                                       dst_usage.stage, dst_usage.access);
    } else {
      auto it = usage_by_ind_.find(ind);
      ++it;
      if (it == usage_by_ind_.end() || !it->second.IsModify()) {
        continue;
      }
      auto src_usage = GetSrcUsage(it->first);
      result[ind] = buffer_.GetBarrier(src_usage.stage, src_usage.access,
                                       it->second.stage, it->second.access);
    }
  }
  return result;
}

Buffer* BufferManager::GetBuffer() {
  return &buffer_;
}

}  // namespace gpu_resources
