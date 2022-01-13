#include "gpu_resources/image.h"

#include "base/base.h"

namespace gpu_resources {

Image::Image(vk::Extent2D extent,
             vk::Format format,
             vk::ImageUsageFlags image_usage)
    : extent_(extent), format_(format) {
  assert(extent.height > 0 && extent.width > 0);
  auto device = base::Base::Get().GetContext().GetDevice();
  image_ = device.createImage(vk::ImageCreateInfo(
      {}, vk::ImageType::e2D, format_, vk::Extent3D(extent_, 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, image_usage,
      vk::SharingMode::eExclusive, {}, {}));
}

Image::Image(Image&& other) noexcept {
  *this = std::move(other);
}

void Image::operator=(Image&& other) noexcept {
  Image tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Image::Swap(Image& other) noexcept {
  std::swap(image_, other.image_);
  std::swap(extent_, other.extent_);
  std::swap(format_, other.format_);
}

vk::Image Image::GetImage() const {
  return image_;
}

vk::BindImageMemoryInfo Image::GetBindMemoryInfo(MemoryBlock memory) const {
  auto requierments = GetMemoryRequierments();
  assert(memory.size >= requierments.size);
  assert(memory.offset % requierments.alignment == 0);
  return vk::BindImageMemoryInfo(image_, memory.memory, memory.offset);
}

vk::MemoryRequirements Image::GetMemoryRequierments() const {
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getImageMemoryRequirements(image_);
}

vk::ImageSubresourceRange Image::GetSubresourceRange() const {
  return vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
}

vk::ImageMemoryBarrier2KHR Image::GetBarrier(
    vk::PipelineStageFlags2KHR src_stage_flags,
    vk::AccessFlags2KHR src_access_flags,
    vk::PipelineStageFlags2KHR dst_stage_flags,
    vk::AccessFlags2KHR dst_access_flags,
    vk::ImageLayout src_layout,
    vk::ImageLayout dst_layout) const {
  return vk::ImageMemoryBarrier2KHR(
      src_stage_flags, src_access_flags, dst_stage_flags, dst_access_flags,
      src_layout, dst_layout, {}, {}, image_, GetSubresourceRange());
}

}  // namespace gpu_resources
