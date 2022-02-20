#pragma once

#include "gpu_resources/logical_buffer.h"
#include "pipeline_handler/binding.h"

namespace render_graph {

class BufferPassBind : public pipeline_handler::Binding {
  gpu_resources::ResourceUsage buffer_usage_;
  vk::BufferUsageFlags buffer_usage_flags_;
  gpu_resources::LogicalBuffer* buffer_ = nullptr;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  BufferPassBind(gpu_resources::ResourceUsage usage,
                 vk::BufferUsageFlags buffer_usage_flags,
                 vk::DescriptorType descriptor_type = {},
                 vk::ShaderStageFlags stage_flags = {});

  void OnResourceBind(uint32_t user_ind, gpu_resources::LogicalBuffer* buffer);
  gpu_resources::LogicalBuffer* GetBoundBuffer() const;
  vk::BufferMemoryBarrier2KHR GetBarrier(uint32_t user_ind) const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
