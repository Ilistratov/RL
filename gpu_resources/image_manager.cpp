#include "gpu_resources/image_manager.h"

#include "base/base.h"

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

ImageManager::ImageManager(vk::Image image,
                           vk::Extent2D extent,
                           vk::Format format,
                           vk::MemoryPropertyFlags memory_properties)
    : extent_(extent), format_(format), memory_properties_(memory_properties) {
  if (image) {
    image_ = Image(image, extent, format);
  }
}

ImageManager ImageManager::CreateStorageImage() {
  auto& swapchain = base::Base::Get().GetSwapchain();
  return ImageManager({}, swapchain.GetExtent(), swapchain.GetFormat(),
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
}

ImageManager ImageManager::CreateSwapchainImage(uint32_t swapchain_image_ind) {
  auto& swapchain = base::Base::Get().GetSwapchain();
  return ImageManager(swapchain.GetImage(swapchain_image_ind),
                      swapchain.GetExtent(), swapchain.GetFormat(),
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
}

ImageManager::ImageManager(ImageManager&& other) noexcept {
  Swap(other);
}
void ImageManager::operator=(ImageManager&& other) noexcept {
  ImageManager tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void ImageManager::Swap(ImageManager& other) noexcept {
  image_.Swap(other.image_);
  std::swap(extent_, other.extent_);
  std::swap(format_, other.format_);
  std::swap(memory_properties_, other.memory_properties_);
}

void ImageManager::CreateImage() {
  if (image_.IsManaged()) {
    assert(!image_.GetImage());
    image_ = Image(extent_, format_, GetAccumulatedUsage());
  }
}

void ImageManager::ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const {
  if (image_.IsManaged()) {
    allocator.AddMemoryBlock(image_.GetMemoryRequierments(),
                             memory_properties_);
  }
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
    if (dst_usage.layout == vk::ImageLayout::eUndefined ||
        dst_usage.layout == vk::ImageLayout::ePreinitialized) {
      dst_usage.layout = src_usage.layout;
    }
    vk::ImageMemoryBarrier2KHR barrier(
        src_usage.stage, src_usage.access, dst_usage.stage, dst_usage.access,
        src_usage.layout, dst_usage.layout, {}, {}, image_.GetImage(),
        image_.GetSubresourceRange());
    result[ind] = barrier;
  }
  return result;
}

Image* ImageManager::GetImage() {
  return &image_;
}

}  // namespace gpu_resources
