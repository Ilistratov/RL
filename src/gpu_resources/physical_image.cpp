#include "gpu_resources/physical_image.h"

#include <stdint.h>
#include <algorithm>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "base/base.h"
#include "gpu_resources/common.h"
#include "utill/error_handling.h"


namespace gpu_resources {

using namespace error_messages;

ImageProperties ImageProperties::Unite(const ImageProperties& lhs,
                                       const ImageProperties& rhs) {
  CHECK(lhs.format == rhs.format);
  return ImageProperties{
      vk::Extent2D(std::max(lhs.extent.width, rhs.extent.width),
                   std::max(lhs.extent.height, rhs.extent.height)),
      lhs.format,
      lhs.memory_flags | rhs.memory_flags,
      lhs.usage_flags | rhs.usage_flags,
  };
}

void PhysicalImage::CreateVkImage() {
  DCHECK(properties_.extent.height > 0 && properties_.extent.width > 0)
      << kErrCantBeEmpty;
  DCHECK(!image_) << kErrAlreadyInitialized;
  auto device = base::Base::Get().GetContext().GetDevice();
  image_ = device.createImage(vk::ImageCreateInfo(
      {}, vk::ImageType::e2D, properties_.format,
      vk::Extent3D(properties_.extent, 1), 1, 1, vk::SampleCountFlagBits::e1,
      vk::ImageTiling::eOptimal, properties_.usage_flags,
      vk::SharingMode::eExclusive, {}, {}));
}

void PhysicalImage::SetDebugName(const std::string& debug_name) const {
  DCHECK(image_) << kErrNotInitialized;
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      image_.objectType, (uint64_t)(VkImage)image_, debug_name.c_str()));
}

void PhysicalImage::RequestMemory(DeviceMemoryAllocator& allocator) {
  DCHECK(image_) << kErrNotInitialized;
  DCHECK(!memory_) << kErrMemoryAlreadyRequested;
  auto device = base::Base::Get().GetContext().GetDevice();
  auto mem_requierments = device.getImageMemoryRequirements(image_);
  memory_ = allocator.RequestMemory(mem_requierments, properties_.memory_flags);
}

vk::BindImageMemoryInfo PhysicalImage::GetBindMemoryInfo() const {
  DCHECK(image_) << kErrNotInitialized;
  DCHECK(memory_) << kErrMemoryNotRequested;
  DCHECK(memory_->memory) << kErrMemoryNotAllocated;
  return vk::BindImageMemoryInfo(image_, memory_->memory, memory_->offset);
}

vk::ImageAspectFlags PhysicalImage::GetAspectFlags() const {
  // add check for depth image when depth images are supported
  return vk::ImageAspectFlagBits::eColor;
}

PhysicalImage::PhysicalImage(uint32_t resource_idx, ImageProperties properties)
    : resource_idx_(resource_idx), properties_(properties) {}

PhysicalImage::PhysicalImage(vk::Image image,
                             vk::Extent2D extent,
                             vk::Format format)
    : resource_idx_(UINT32_MAX),
      image_(image),
      properties_(ImageProperties{extent, format, {}, {}}) {}

PhysicalImage::PhysicalImage(PhysicalImage&& other) noexcept {
  Swap(other);
}

void PhysicalImage::operator=(PhysicalImage&& other) noexcept {
  PhysicalImage tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void PhysicalImage::Swap(PhysicalImage& other) noexcept {
  std::swap(resource_idx_, other.resource_idx_);
  std::swap(image_, other.image_);
  std::swap(properties_, other.properties_);
  std::swap(image_view_, other.image_view_);
  std::swap(memory_, other.memory_);
}

PhysicalImage::~PhysicalImage() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyImageView(image_view_);
  device.destroyImage(image_);
}

vk::Image PhysicalImage::Release() {
  PhysicalImage tmp;
  Swap(tmp);
  auto image = tmp.image_;
  tmp.image_ = vk::Image{};
  return image;
}

uint32_t PhysicalImage::GetIdx() const {
  return resource_idx_;
}

vk::Image PhysicalImage::GetImage() const {
  return image_;
}

vk::Extent2D PhysicalImage::GetExtent() const {
  return properties_.extent;
}

vk::Format PhysicalImage::GetFormat() const {
  return properties_.format;
}

vk::ImageSubresourceRange PhysicalImage::GetSubresourceRange() const {
  return vk::ImageSubresourceRange(GetAspectFlags(), 0, 1, 0, 1);
}

vk::ImageSubresourceLayers PhysicalImage::GetSubresourceLayers() const {
  return vk::ImageSubresourceLayers(GetAspectFlags(), 0, 0, 1);
}

vk::ImageMemoryBarrier2KHR PhysicalImage::GenerateBarrier(
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

void PhysicalImage::CreateImageView() {
  if (image_view_) {
    return;
  }
  DCHECK(image_) << kErrNotInitialized;
  auto device = base::Base::Get().GetContext().GetDevice();
  image_view_ = device.createImageView(vk::ImageViewCreateInfo(
      {}, image_, vk::ImageViewType::e2D, properties_.format,
      vk::ComponentMapping{
          vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
          vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
      GetSubresourceRange()));
}

vk::ImageView PhysicalImage::GetImageView() const {
  return image_view_;
}

}  // namespace gpu_resources
