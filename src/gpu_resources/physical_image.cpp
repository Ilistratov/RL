#include "gpu_resources/physical_image.h"

#include "base/base.h"

namespace gpu_resources {

PhysicalImage::PhysicalImage(vk::Extent2D extent,
                             vk::Format format,
                             vk::ImageUsageFlags image_usage)
    : extent_(extent), format_(format), is_managed_(true) {
  assert(extent.height > 0 && extent.width > 0);
  auto device = base::Base::Get().GetContext().GetDevice();
  image_ = device.createImage(vk::ImageCreateInfo(
      {}, vk::ImageType::e2D, format_, vk::Extent3D(extent_, 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, image_usage,
      vk::SharingMode::eExclusive, {}, {}));
}

PhysicalImage::PhysicalImage(vk::Image image,
                             vk::Extent2D extent,
                             vk::Format format)
    : image_(image), extent_(extent), format_(format), is_managed_(false) {}

PhysicalImage::PhysicalImage(PhysicalImage&& other) noexcept {
  Swap(other);
}

void PhysicalImage::operator=(PhysicalImage&& other) noexcept {
  PhysicalImage tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void PhysicalImage::Swap(PhysicalImage& other) noexcept {
  std::swap(image_, other.image_);
  std::swap(extent_, other.extent_);
  std::swap(format_, other.format_);
  std::swap(is_managed_, other.is_managed_);
}

vk::Image PhysicalImage::GetImage() const {
  return image_;
}

vk::Extent2D PhysicalImage::GetExtent() const {
  return extent_;
}

vk::Format PhysicalImage::GetFormat() const {
  return format_;
}

bool PhysicalImage::IsManaged() const {
  return is_managed_;
}

vk::MemoryRequirements PhysicalImage::GetMemoryRequierments() const {
  if (!is_managed_) {
    return {};
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.getImageMemoryRequirements(image_);
}

vk::BindImageMemoryInfo PhysicalImage::GetBindMemoryInfo(
    MemoryBlock memory_block) const {
  assert(is_managed_);
  assert(image_);
  assert(memory_block.memory);
  return vk::BindImageMemoryInfo(image_, memory_block.memory,
                                 memory_block.offset);
}

vk::ImageSubresourceRange PhysicalImage::GetSubresourceRange() const {
  return vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
}

vk::ImageSubresourceLayers PhysicalImage::GetSubresourceLayers() const {
  return vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
}

vk::ImageMemoryBarrier2KHR PhysicalImage::GetBarrier(
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

void PhysicalImage::SetDebugName(const std::string& debug_name) const {
  assert(image_);
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      image_.objectType, (uint64_t)(VkImage)image_, debug_name.c_str()));
}

PhysicalImage::~PhysicalImage() {
  if (is_managed_) {
    auto device = base::Base::Get().GetContext().GetDevice();
    device.destroyImage(image_);
  }
}

}  // namespace gpu_resources
