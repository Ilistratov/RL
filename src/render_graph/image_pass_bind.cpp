#include "render_graph/image_pass_bind.h"

#include "utill/error_handling.h"

namespace render_graph {

ImagePassBind::ImagePassBind(gpu_resources::ResourceUsage usage,
                             vk::ImageUsageFlags image_usage_flags,
                             vk::DescriptorType descriptor_type,
                             vk::ShaderStageFlags stage_flags)
    : image_usage_(usage),
      image_usage_flags_(image_usage_flags),
      descriptor_type_(descriptor_type),
      descriptor_stage_flags_(stage_flags) {}

void ImagePassBind::OnResourceBind(uint32_t user_ind,
                                   gpu_resources::LogicalImage* image) {
  DCHECK(!image_) << "Resource already bound";
  DCHECK(image) << "Cant bind null";
  image_ = image;
  image_->AddUsage(user_ind, image_usage_, image_usage_flags_);
}

gpu_resources::LogicalImage* ImagePassBind::GetBoundImage() const {
  return image_;
}

vk::ImageMemoryBarrier2KHR ImagePassBind::GetBarrier(uint32_t user_ind) const {
  DCHECK(image_) << "Resource must be bound to use this method";
  return image_->GetPostPassBarrier(user_ind);
}

vk::DescriptorSetLayoutBinding ImagePassBind::GetVkBinding() const noexcept {
  return vk::DescriptorSetLayoutBinding(0, descriptor_type_, 1,
                                        descriptor_stage_flags_, {});
}

pipeline_handler::Write ImagePassBind::GetWrite() const noexcept {
  DCHECK(image_) << "Resource must be bound to use this method";
  if (!image_->GetPhysicalImage().GetImageView()) {
    image_->GetPhysicalImage().CreateImageView();
  }

  return pipeline_handler::Write{
      0,
      descriptor_type_,
      {vk::DescriptorImageInfo({}, image_->GetPhysicalImage().GetImageView(),
                               image_usage_.layout)},
      {}};
}

}  // namespace render_graph
