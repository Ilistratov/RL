#include "gpu_resources/logical_image.h"

#include "base/base.h"

namespace gpu_resources {

LogicalImage::LogicalImage(vk::Extent2D extent,
                           vk::Format format,
                           vk::MemoryPropertyFlags memory_flags)
    : extent_(extent), format_(format), memory_flags_(memory_flags) {}

LogicalImage::LogicalImage(LogicalImage&& other) noexcept {
  Swap(other);
}

void LogicalImage::operator=(LogicalImage&& other) noexcept {
  LogicalImage tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void LogicalImage::Swap(LogicalImage& other) noexcept {
  image_.Swap(other.image_);
  std::swap(access_manager_, other.access_manager_);
  std::swap(extent_, other.extent_);
  std::swap(format_, other.format_);
  std::swap(memory_flags_, other.memory_flags_);
  std::swap(usage_flags_, other.usage_flags_);
  std::swap(memory_, other.memory_);
}

LogicalImage LogicalImage::CreateStorageImage(vk::Extent2D extent) {
  if (extent == vk::Extent2D(0, 0)) {
    extent = base::Base::Get().GetSwapchain().GetExtent();
  }
  vk::Format format = base::Base::Get().GetSwapchain().GetFormat();
  return LogicalImage(extent, format, vk::MemoryPropertyFlagBits::eDeviceLocal);
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
  memory_ =
      allocator.RequestMemory(image_.GetMemoryRequierments(), memory_flags_);
}

vk::BindImageMemoryInfo LogicalImage::GetBindMemoryInfo() const {
  assert(memory_);
  return image_.GetBindMemoryInfo(*memory_);
}

PhysicalImage& LogicalImage::GetPhysicalImage() {
  return image_;
}

void LogicalImage::AddUsage(uint32_t user_ind,
                            ResourceUsage usage,
                            vk::ImageUsageFlags image_usage_flags) {
  access_manager_.AddUsage(user_ind, usage);
  usage_flags_ |= image_usage_flags;
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

vk::ImageMemoryBarrier2KHR LogicalImage::GetInitBarrier() const {
  auto dst_usage = access_manager_.GetFirstUsage();
  return image_.GetBarrier(vk::PipelineStageFlagBits2KHR::eBottomOfPipe, {},
                           dst_usage.stage, dst_usage.access,
                           vk::ImageLayout::eUndefined, dst_usage.layout);
}

}  // namespace gpu_resources
