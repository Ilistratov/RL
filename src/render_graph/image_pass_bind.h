#pragma once

#include "gpu_resources/logical_image.h"
#include "pipeline_handler/binding.h"

namespace render_graph {

class ImagePassBind : public pipeline_handler::Binding {
  gpu_resources::ResourceUsage image_usage_;
  gpu_resources::LogicalImage* image_ = nullptr;
  uint32_t user_ind_;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  ImagePassBind(gpu_resources::ResourceUsage usage,
                uint32_t user_ind,
                vk::DescriptorType descriptor_type = {},
                vk::ShaderStageFlags stage_flags = {});

  void OnResourceBind(gpu_resources::LogicalImage* image);
  gpu_resources::LogicalImage* GetBoundImage() const;
  vk::ImageMemoryBarrier2KHR GetBarrier() const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
