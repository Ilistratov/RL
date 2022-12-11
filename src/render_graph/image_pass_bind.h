#pragma once

#include "gpu_resources/image.h"
#include "pipeline_handler/descriptor_binding.h"

namespace render_graph {

class ImagePassBind : public pipeline_handler::DescriptorBinding {
  gpu_resources::ResourceAccess image_usage_;
  vk::ImageUsageFlags image_usage_flags_;
  gpu_resources::Image* image_ = nullptr;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  ImagePassBind() = default;
  ImagePassBind(gpu_resources::ResourceAccess usage,
                vk::ImageUsageFlags image_usage_flags,
                vk::DescriptorType descriptor_type = {},
                vk::ShaderStageFlags stage_flags = {});
  static ImagePassBind ComputeRenderTarget(vk::AccessFlagBits2KHR access_flags);

  void OnResourceBind(uint32_t user_ind, gpu_resources::Image* image);
  gpu_resources::Image* GetBoundImage() const;
  vk::ImageMemoryBarrier2KHR GetBarrier(uint32_t user_ind) const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
