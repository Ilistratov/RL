#pragma once

#include "gpu_resources/logical_image.h"
#include "pipeline_handler/binding.h"

namespace render_graph {

class ImagePassBind : public pipeline_handler::Binding {
  gpu_resources::ResourceUsage image_usage_;
  gpu_resources::LogicalImage* image_ = nullptr;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  ImagePassBind(gpu_resources::ResourceUsage usage,
                vk::DescriptorType descriptor_type = {},
                vk::ShaderStageFlags stage_flags = {});

  void OnResourceBind(uint32_t user_ind, gpu_resources::LogicalImage* image);
  gpu_resources::LogicalImage* GetBoundImage() const;
  vk::ImageMemoryBarrier2KHR GetBarrier(uint32_t user_ind) const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
