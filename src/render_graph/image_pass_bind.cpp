#include "render_graph/image_pass_bind.h"

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
  assert(!image_);
  assert(image);
  image_ = image;
  image_->AddUsage(user_ind, image_usage_, image_usage_flags_);
}

gpu_resources::LogicalImage* ImagePassBind::GetBoundImage() const {
  return image_;
}

vk::ImageMemoryBarrier2KHR ImagePassBind::GetBarrier(uint32_t user_ind) const {
  assert(image_);
  return image_->GetPostPassBarrier(user_ind);
}

vk::DescriptorSetLayoutBinding ImagePassBind::GetVkBinding() const noexcept {
  return vk::DescriptorSetLayoutBinding(0, descriptor_type_, 1,
                                        descriptor_stage_flags_, {});
}

pipeline_handler::Write ImagePassBind::GetWrite() const noexcept {
  assert(image_);
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
