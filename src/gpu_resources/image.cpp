#include "gpu_resources/image.h"

#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "gpu_resources/common.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

Image::Image(ImageProperties properties, PassAccessSyncronizer* syncronizer)
    : syncronizer_(syncronizer), required_properties_(properties) {
  DCHECK(syncronizer_) << kErrSyncronizerNotProvided;
}

void Image::DeclareAccess(ResourceAccess access, uint32_t pass_idx) const {
  DCHECK(syncronizer_) << kErrSyncronizerNotProvided;
  syncronizer_->AddAccess(image_, access, pass_idx);
}

vk::ImageView Image::GetImageView() const noexcept {
  DCHECK(image_) << kErrResourceIsNull;
  return image_->GetImageView();
}

void Image::CreateImageView() {
  DCHECK(image_) << kErrResourceIsNull;
  image_->CreateImageView();
}

void Image::RecordBlit(vk::CommandBuffer cmd,
                       const Image& src,
                       const Image& dst) {
  DCHECK(src.image_) << "src image: " << kErrResourceIsNull;
  DCHECK(src.image_->GetImage()) << "src image: " << kErrNotInitialized;
  DCHECK(dst.image_) << "dst image: " << kErrResourceIsNull;
  DCHECK(dst.image_->GetImage()) << "dst image: " << kErrNotInitialized;
  vk::Extent2D src_extent = src.image_->GetExtent();
  vk::Extent2D dst_extent = dst.image_->GetExtent();
  vk::ImageBlit2KHR region(
      src.image_->GetSubresourceLayers(),
      {vk::Offset3D(0, 0, 0),
       vk::Offset3D(src_extent.width, src_extent.height, 1)},
      dst.image_->GetSubresourceLayers(),
      {vk::Offset3D(0, 0, 0),
       vk::Offset3D(dst_extent.width, dst_extent.height, 1)});
  vk::BlitImageInfo2KHR blit_info(
      src.image_->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
      dst.image_->GetImage(), vk::ImageLayout::eTransferDstOptimal, region);
  cmd.blitImage2KHR(blit_info);
}

}  // namespace gpu_resources
