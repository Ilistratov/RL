#include "render_graph/image_pass_bind.h"

namespace render_graph {

ImagePassBind::ImagePassBind(gpu_resources::ResourceUsage usage,
                             uint32_t user_ind,
                             vk::DescriptorType descriptor_type,
                             vk::ShaderStageFlags stage_flags)
    : image_usage_(usage),
      user_ind_(user_ind),
      descriptor_type_(descriptor_type),
      descriptor_stage_flags_(stage_flags) {}

void ImagePassBind::OnResourceBind(gpu_resources::LogicalImage* image) {
  assert(!image_);
  assert(image);
  image_ = image;
}

gpu_resources::LogicalImage* ImagePassBind::GetBoundImage() const {
  return image_;
}

vk::ImageMemoryBarrier2KHR ImagePassBind::GetBarrier() const {
  assert(image_);
  return image_->GetPostPassBarrier(user_ind_);
}

vk::DescriptorSetLayoutBinding ImagePassBind::GetVkBinding() const noexcept {
  return vk::DescriptorSetLayoutBinding(0, descriptor_type_,
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