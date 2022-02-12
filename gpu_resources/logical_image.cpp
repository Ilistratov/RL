#include "gpu_resources/logical_image.h"

#include "base/base.h"

namespace gpu_resources {

LogicalImage::LogicalImage(vk::Image image,
                           vk::Extent2D extent,
                           vk::Format format,
                           vk::MemoryPropertyFlags memory_flags)
    : extent_(extent), format_(format), memory_flags_(memory_flags) {
  if (image) {
    image_ = PhysicalImage(image, extent_, format_);
  }
}

LogicalImage LogicalImage::CreateStorageImage(vk::Extent2D extent) {
  if (extent == vk::Extent2D(0, 0)) {
    extent = base::Base::Get().GetSwapchain().GetExtent();
  }
  vk::Format format = base::Base::Get().GetSwapchain().GetFormat();
  return LogicalImage({}, extent, format,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
}

LogicalImage LogicalImage::CreateSwapchainImage(uint32_t swapchain_image_ind) {
  auto& swapchain = base::Base::Get().GetSwapchain();
  return LogicalImage(swapchain.GetImage(swapchain_image_ind),
                      swapchain.GetExtent(), swapchain.GetFormat(), {});
}

void LogicalImage::Create() {
  if (!image_.GetImage()) {
    image_ = PhysicalImage(extent_, format_, usage_flags_);
  }
}

void LogicalImage::SetDebugName(const std::string& debug_name) const {
  image_.SetDebugName(debug_name);
}

void LogicalImage::RequestMemory(DeviceMemoryAllocator& allocator) {
  assert(image_.GetImage());
  if (image_.IsManaged()) {
    memory_ =
        allocator.RequestMemory(image_.GetMemoryRequierments(), memory_flags_);
  }
}

vk::BindImageMemoryInfo LogicalImage::GetBindMemoryInfo() const {
  if (image_.IsManaged()) {
    assert(memory_);
    return image_.GetBindMemoryInfo(*memory_);
  }
  return {};
}

vk::ImageMemoryBarrier2KHR LogicalImage::GetPostPassBarrier(uint32_t user_ind) {
  auto [src_usage, dst_usage] = access_manager_.GetUserDeps(user_ind);
  if (src_usage.stage == vk::PipelineStageFlagBits2KHR::eNone &&
      dst_usage.stage == vk::PipelineStageFlagBits2KHR::eNone) {
    return {};
  }
  return image_.GetBarrier(src_usage.stage, src_usage.access, dst_usage.stage,
                           dst_usage.access, src_usage.layout,
                           dst_usage.layout);
}

}  // namespace gpu_resources
