#include "gpu_resources/image_manager.h"

namespace gpu_resources {

ImageManager::Usage& ImageManager::Usage::operator|=(
    const ImageManager::Usage& other) {
  access |= other.access;
  stage |= other.stage;
  usage |= other.usage;
  assert(layout == other.layout);
  return *this;
}

bool ImageManager::Usage::IsModify() const {
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

vk::ImageUsageFlags ImageManager::GetAccumulatedUsage() const {
  vk::ImageUsageFlags result;
  for (auto [ind, usage] : usage_by_ind_) {
    result |= usage.usage;
  }
  return result;
}

bool ImageManager::IsLayoutTransitionNeeded(uint32_t user_ind) const {
  auto it = usage_by_ind_.find(user_ind);
  assert(it != usage_by_ind_.end());
  auto src_layout = it->second.layout;
  ++it;
  if (it == usage_by_ind_.end()) {
    it = usage_by_ind_.begin();
  }
  auto dst_layout = it->second.layout;
  return src_layout != dst_layout;
}

std::map<uint32_t, ImageManager::Usage>::const_iterator
ImageManager::LoopedNext(
    std::map<uint32_t, ImageManager::Usage>::const_iterator it) const {
  ++it;
  if (it == usage_by_ind_.end()) {
    it = usage_by_ind_.begin();
  }
  return it;
}

std::map<uint32_t, ImageManager::Usage>::const_iterator
ImageManager::LoopedPrev(
    std::map<uint32_t, ImageManager::Usage>::const_iterator it) const {
  if (it == usage_by_ind_.begin()) {
    it = usage_by_ind_.end();
  }
  --it;
  return it;
}

ImageManager::Usage ImageManager::GetDstUsage(uint32_t src_user_ind) const {
  auto it = usage_by_ind_.find(src_user_ind);
  assert(it != usage_by_ind_.end());
  it = LoopedNext(it);
  Usage result;
  result = it->second;
  while (!it->second.IsModify() && !IsLayoutTransitionNeeded(it->first)) {
    result |= it->second;
    if (it->first == src_user_ind) {
      break;
    }
    it = LoopedNext(it);
  }
  return result;
}

ImageManager::Usage ImageManager::GetSrcUsage(uint32_t dst_user_ind) const {
  auto it = usage_by_ind_.find(dst_user_ind);
  assert(it != usage_by_ind_.end());
  it = LoopedPrev(it);
  Usage result;
  result = it->second;
  it = LoopedPrev(it);
  while (!it->second.IsModify() && !IsLayoutTransitionNeeded(it->first)) {
    result |= it->second;
    if (it->first == dst_user_ind) {
      break;
    }
    it = LoopedPrev(it);
  }
  return result;
}

ImageManager::ImageManager(vk::Extent2D extent,
                           vk::Format format,
                           vk::MemoryPropertyFlags memory_properties)
    : extent_(extent), format_(format), memory_properties_(memory_properties) {}

void ImageManager::AddUsage(uint32_t user_ind, ImageManager::Usage usage) {
  assert(!usage_by_ind_.contains(user_ind));
  usage_by_ind_[user_ind] = usage;
}

void ImageManager::CreateImage() {
  assert(image_.GetImage());
  image_ = Image(extent_, format_, GetAccumulatedUsage());
}

void ImageManager::ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const {
  allocator.AddMemoryBlock(image_.GetMemoryRequierments(), memory_properties_);
}

vk::BindImageMemoryInfo ImageManager::GetBindMemoryInfo(
    DeviceMemoryAllocator& allocator) const {
  auto memory_block = allocator.GetMemoryBlock(image_.GetMemoryRequierments(),
                                               memory_properties_);
  return image_.GetBindMemoryInfo(memory_block);
}

std::map<uint32_t, vk::ImageMemoryBarrier2KHR> ImageManager::GetBarriers()
    const {
  std::map<uint32_t, vk::ImageMemoryBarrier2KHR> result;
  for (auto [ind, usage] : usage_by_ind_) {
    if (usage.IsModify()) {
      auto dst_usage = GetDstUsage(ind);
      result[ind] = vk::ImageMemoryBarrier2KHR(
          usage.stage, usage.access, dst_usage.stage, dst_usage.access,
          usage.layout, dst_usage.layout, {}, {}, image_.GetImage(),
          image_.GetSubresourceRange());
    } else {
      auto it = usage_by_ind_.find(ind);
      it = LoopedNext(it);
      if (!it->second.IsModify() && !IsLayoutTransitionNeeded(ind)) {
        continue;
      }
      auto src_usage = GetSrcUsage(it->first);
      auto dst_usage = GetDstUsage(ind);
      result[ind] = vk::ImageMemoryBarrier2KHR(
          src_usage.stage, src_usage.access, dst_usage.stage, dst_usage.access,
          src_usage.layout, dst_usage.layout, {}, {}, image_.GetImage(),
          image_.GetSubresourceRange());
    }
  }
  return result;
}

}  // namespace gpu_resources
