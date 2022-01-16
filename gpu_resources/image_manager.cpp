#include "gpu_resources/image_manager.h"

namespace gpu_resources {

ImageUsage& ImageUsage::operator|=(ImageUsage other) {
  access |= other.access;
  stage |= other.stage;
  usage |= other.usage;
  assert(layout == other.layout);
  return *this;
}

bool ImageUsage::IsModify() const {
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

bool ImageUsage::IsDependencyNeeded(ImageUsage other) const {
  return IsModify() || other.IsModify() || layout != other.layout;
}

vk::ImageUsageFlags ImageManager::GetAccumulatedUsage() const {
  vk::ImageUsageFlags result;
  for (auto [ind, usage] : usage_by_ind_) {
    result |= usage.usage;
  }
  return result;
}

ImageManager::ImageManager(vk::Extent2D extent,
                           vk::Format format,
                           vk::MemoryPropertyFlags memory_properties)
    : extent_(extent), format_(format), memory_properties_(memory_properties) {}

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
    auto src_it = usage_by_ind_.find(ind);
    auto dst_it = LoopedNext(src_it);
    if (!usage.IsDependencyNeeded(dst_it->second)) {
      continue;
    }
    auto [src_usage, dst_usage] = GetUsageForBarrier(ind);
    vk::ImageMemoryBarrier2KHR barrier(
        src_usage.stage, src_usage.access, dst_usage.stage, dst_usage.access,
        src_usage.layout, dst_usage.layout, {}, {}, image_.GetImage(),
        image_.GetSubresourceRange());
    result[ind] = barrier;
  }
  return result;
}

}  // namespace gpu_resources
