#include "render_graph/buffer_pass_bind.h"

#include "utill/error_handling.h"

namespace render_graph {

BufferPassBind::BufferPassBind(gpu_resources::ResourceUsage usage,
                               vk::BufferUsageFlags buffer_usage_flags,
                               vk::DescriptorType descriptor_type,
                               vk::ShaderStageFlags stage_flags)
    : buffer_usage_(usage),
      buffer_usage_flags_(buffer_usage_flags),
      descriptor_type_(descriptor_type),
      descriptor_stage_flags_(stage_flags) {}

BufferPassBind BufferPassBind::ComputeStorageBuffer(
    vk::AccessFlags2KHR access_flags) {
  gpu_resources::ResourceUsage usage;
  usage.access = access_flags;
  usage.stage = vk::PipelineStageFlagBits2KHR::eComputeShader;
  return BufferPassBind(usage, vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::DescriptorType::eStorageBuffer,
                        vk::ShaderStageFlagBits::eCompute);
}

BufferPassBind BufferPassBind::UniformBuffer(
    vk::PipelineStageFlags2KHR pipeline_stage,
    vk::ShaderStageFlags shader_stage) {
  gpu_resources::ResourceUsage usage;
  usage.access = vk::AccessFlagBits2KHR::eUniformRead;
  usage.stage = pipeline_stage;
  return BufferPassBind(usage, vk::BufferUsageFlagBits::eUniformBuffer,
                        vk::DescriptorType::eUniformBuffer, shader_stage);
}

BufferPassBind BufferPassBind::TransferSrcBuffer() {
  gpu_resources::ResourceUsage usage;
  usage.access = vk::AccessFlagBits2KHR::eTransferRead;
  usage.stage = vk::PipelineStageFlagBits2KHR::eTransfer;
  return BufferPassBind(usage, vk::BufferUsageFlagBits::eTransferSrc, {}, {});
}

BufferPassBind BufferPassBind::TransferDstBuffer() {
  gpu_resources::ResourceUsage usage;
  usage.access = vk::AccessFlagBits2KHR::eTransferWrite;
  usage.stage = vk::PipelineStageFlagBits2KHR::eTransfer;
  return BufferPassBind(usage, vk::BufferUsageFlagBits::eTransferDst, {}, {});
}

void BufferPassBind::OnResourceBind(uint32_t user_ind,
                                    gpu_resources::LogicalBuffer* buffer) {
  DCHECK(!buffer_) << "Resource already bound";
  DCHECK(buffer) << "Can't bind null";
  buffer_ = buffer;
  buffer_->AddUsage(user_ind, buffer_usage_, buffer_usage_flags_);
}

gpu_resources::LogicalBuffer* BufferPassBind::GetBoundBuffer() const {
  return buffer_;
}

vk::BufferMemoryBarrier2KHR BufferPassBind::GetBarrier(
    uint32_t user_ind) const {
  DCHECK(buffer_) << "Resource must be bound to use this method";
  return buffer_->GetPostPassBarrier(user_ind);
}

vk::DescriptorSetLayoutBinding BufferPassBind::GetVkBinding() const noexcept {
  return vk::DescriptorSetLayoutBinding(0, descriptor_type_, 1,
                                        descriptor_stage_flags_, {});
}

pipeline_handler::Write BufferPassBind::GetWrite() const noexcept {
  DCHECK(buffer_) << "Resource must be bound to use this method";
  return pipeline_handler::Write{
      0,
      descriptor_type_,
      {},
      {vk::DescriptorBufferInfo(buffer_->GetPhysicalBuffer().GetBuffer(), 0,
                                buffer_->GetPhysicalBuffer().GetSize())}};
}

}  // namespace render_graph
