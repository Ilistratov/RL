#include "gpu_resources/image.h"

#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "gpu_resources/common.h"
#include "gpu_resources/physical_image.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

Image::Image(ImageProperties properties, PassAccessSyncronizer* syncronizer)
    : syncronizer_(syncronizer), required_properties_(properties) {
  DCHECK(syncronizer_) << kErrSyncronizerNotProvided;
}

void Image::RequireProperties(ImageProperties properties) {
  required_properties_ =
      ImageProperties::Unite(required_properties_, properties);
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

PhysicalImage* Image::GetImage() {
  return image_;
}

}  // namespace gpu_resources
