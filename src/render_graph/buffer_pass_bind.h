#pragma once

#include "gpu_resources/logical_buffer.h"
#include "pipeline_handler/binding.h"

namespace render_graph {

class BufferPassBind : public pipeline_handler::Binding {
  gpu_resources::ResourceUsage buffer_usage_;
  gpu_resources::LogicalBuffer* buffer_ = nullptr;
  uint32_t user_ind_;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  BufferPassBind(gpu_resources::ResourceUsage usage,
                 uint32_t user_ind,
                 vk::DescriptorType descriptor_type = {},
                 vk::ShaderStageFlags stage_flags = {});

  void OnResourceBind(gpu_resources::LogicalBuffer* buffer);
  gpu_resources::LogicalBuffer* GetBoundBuffer() const;
  vk::BufferMemoryBarrier2KHR GetBarrier() const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
