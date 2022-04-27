#include "gpu_resources/image.h"

#include "base/base.h"
#include "utill/error_handling.h"

namespace gpu_resources {

Image::Image(vk::Extent2D extent,
             vk::Format format,
             vk::MemoryPropertyFlags memory_flags)
    : extent_(extent), format_(format), memory_flags_(memory_flags) {}

void Image::AddUsage(uint32_t user_ind,
                     ResourceUsage usage,
                     vk::ImageUsageFlags image_usage_flags) {
  access_manager_.AddUsage(user_ind, usage);
  usage_flags_ |= image_usage_flags;
}

void Image::CreateVkImage() {
  DCHECK(extent_.height > 0 && extent_.width > 0) << "Invalid image extent";
  DCHECK(!image_) << "VkImage already created";
  auto device = base::Base::Get().GetContext().GetDevice();
  image_ = device.createImage(vk::ImageCreateInfo(
      {}, vk::ImageType::e2D, format_, vk::Extent3D(extent_, 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage_flags_,
      vk::SharingMode::eExclusive, {}, {}));
}

void Image::SetDebugName(const std::string& debug_name) const {
  DCHECK(image_) << "Resource must be created to use this method";
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      image_.objectType, (uint64_t)(VkImage)image_, debug_name.c_str()));
}

void Image::RequestMemory(DeviceMemoryAllocator& allocator) {
  DCHECK(image_) << "VkImage must be created to use this method";
  auto device = base::Base::Get().GetContext().GetDevice();
  auto mem_requierments = device.getImageMemoryRequirements(image_);
  memory_ = allocator.RequestMemory(mem_requierments, memory_flags_);
}

vk::BindImageMemoryInfo Image::GetBindMemoryInfo() const {
  DCHECK(image_) << "VkImage must be created to use this method";
  DCHECK(memory_) << "Memory must be requested to use this method";
  DCHECK(memory_->memory)
      << "Requested memory must be allocated to use this method";
  return vk::BindImageMemoryInfo(image_, memory_->memory, memory_->offset);
}

vk::ImageAspectFlags Image::GetAspectFlags() const {
  // add check for depth image when depth images arrive
  return vk::ImageAspectFlagBits::eColor;
}

Image::Image(Image&& other) noexcept {
  Swap(other);
}
void Image::operator=(Image&& other) noexcept {
  Image tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Image::Swap(Image& other) noexcept {
  std::swap(access_manager_, other.access_manager_);
  std::swap(image_, other.image_);
  std::swap(extent_, other.extent_);
  std::swap(format_, other.format_);
  std::swap(image_view_, other.image_view_);
  std::swap(memory_flags_, other.memory_flags_);
  std::swap(usage_flags_, other.usage_flags_);
  std::swap(memory_, other.memory_);
}

Image ::~Image() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyImageView(image_view_);
  device.destroyImage(image_);
}

Image::Image(vk::Image image, vk::Extent2D extent, vk::Format format)
    : image_(image), extent_(extent), format_(format) {}

vk::Image Image::Release() {
  Image tmp;
  Swap(tmp);
  auto image = tmp.image_;
  tmp.image_ = vk::Image{};
  return image;
}

vk::Image Image::GetImage() const {
  return image_;
}

vk::Extent2D Image::GetExtent() const {
  return extent_;
}

vk::Format Image::GetFormat() const {
  return format_;
}

vk::ImageSubresourceRange Image::GetSubresourceRange() const {
  return vk::ImageSubresourceRange(GetAspectFlags(), 0, 1, 0, 1);
}

vk::ImageSubresourceLayers Image::GetSubresourceLayers() const {
  return vk::ImageSubresourceLayers(GetAspectFlags(), 0, 0, 1);
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

vk::ImageMemoryBarrier2KHR Image::GetPostPassBarrier(uint32_t user_ind) {
  auto [src_usage, dst_usage] = access_manager_.GetUserDeps(user_ind);
  if (src_usage.stage == vk::PipelineStageFlagBits2KHR::eNone &&
      dst_usage.stage == vk::PipelineStageFlagBits2KHR::eNone) {
    return {};
  }
  return GetBarrier(src_usage.stage, src_usage.access, dst_usage.stage,
                    dst_usage.access, src_usage.layout, dst_usage.layout);
}

vk::ImageMemoryBarrier2KHR Image::GetInitBarrier() const {
  auto dst_usage = access_manager_.GetFirstUsage();
  return GetBarrier(vk::PipelineStageFlagBits2KHR::eBottomOfPipe, {},
                    dst_usage.stage, dst_usage.access,
                    vk::ImageLayout::eUndefined, dst_usage.layout);
}

void Image::CreateImageView() {
  if (image_view_) {
    return;
  }
  DCHECK(image_) << "VkImage must be created to use this method";
  auto device = base::Base::Get().GetContext().GetDevice();
  image_view_ = device.createImageView(vk::ImageViewCreateInfo(
      {}, image_, vk::ImageViewType::e2D, format_,
      vk::ComponentMapping{
          vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
          vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
      GetSubresourceRange()));
}

vk::ImageView Image::GetImageView() const {
  return image_view_;
}

void Image::RecordBlit(vk::CommandBuffer cmd,
                       const Image& src,
                       const Image& dst) {
  DCHECK(src.image_) << "src image must be created to use this method";
  DCHECK(dst.image_) << "dst image must be created to use this method";
  vk::ImageBlit2KHR region(
      src.GetSubresourceLayers(),
      {vk::Offset3D(0, 0, 0),
       vk::Offset3D(src.GetExtent().width, src.GetExtent().height, 1)},
      dst.GetSubresourceLayers(),
      {vk::Offset3D(0, 0, 0),
       vk::Offset3D(dst.GetExtent().width, dst.GetExtent().height, 1)});
  vk::BlitImageInfo2KHR blit_info(
      src.GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst.GetImage(),
      vk::ImageLayout::eTransferDstOptimal, region);
  cmd.blitImage2KHR(blit_info);
}

}