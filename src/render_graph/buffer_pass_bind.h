#pragma once

#include "gpu_resources/logical_buffer.h"
#include "pipeline_handler/descriptor_binding.h"

namespace render_graph {

class BufferPassBind : public pipeline_handler::DescriptorBinding {
  gpu_resources::ResourceUsage buffer_usage_;
  vk::BufferUsageFlags buffer_usage_flags_;
  gpu_resources::LogicalBuffer* buffer_ = nullptr;
  vk::DescriptorType descriptor_type_;
  vk::ShaderStageFlags descriptor_stage_flags_;

 public:
  BufferPassBind() = default;
  BufferPassBind(gpu_resources::ResourceUsage usage,
                 vk::BufferUsageFlags buffer_usage_flags,
                 vk::DescriptorType descriptor_type = {},
                 vk::ShaderStageFlags stage_flags = {});
  static BufferPassBind ComputeStorageBuffer(vk::AccessFlags2KHR access_flags);
  static BufferPassBind UniformBuffer(vk::PipelineStageFlags2KHR pipeline_stage,
                                      vk::ShaderStageFlags shader_stage = {});
  static BufferPassBind TransferSrcBuffer();
  static BufferPassBind TransferDstBuffer();

  void OnResourceBind(uint32_t user_ind, gpu_resources::LogicalBuffer* buffer);
  gpu_resources::LogicalBuffer* GetBoundBuffer() const;
  vk::BufferMemoryBarrier2KHR GetBarrier(uint32_t user_ind) const;

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  pipeline_handler::Write GetWrite() const noexcept override;
};

}  // namespace render_graph
