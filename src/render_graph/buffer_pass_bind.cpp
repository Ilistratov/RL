#include "render_graph/buffer_pass_bind.h"

namespace render_graph {

BufferPassBind::BufferPassBind(gpu_resources::ResourceUsage usage,
                               uint32_t user_ind,
                               vk::DescriptorType descriptor_type,
                               vk::ShaderStageFlags stage_flags)
    : buffer_usage_(usage),
      user_ind_(user_ind),
      descriptor_type_(descriptor_type),
      descriptor_stage_flags_(stage_flags) {}

void BufferPassBind::OnResourceBind(gpu_resources::LogicalBuffer* buffer) {
  assert(!buffer_);
  assert(buffer);
  buffer_ = buffer;
}

gpu_resources::LogicalBuffer* BufferPassBind::GetBoundBuffer() const {
  return buffer_;
}

vk::BufferMemoryBarrier2KHR BufferPassBind::GetBarrier() const {
  assert(buffer_);
  return buffer_->GetPostPassBarrier(user_ind_);
}

vk::DescriptorSetLayoutBinding BufferPassBind::GetVkBinding() const noexcept {
  return vk::DescriptorSetLayoutBinding(0, descriptor_type_,
                                        descriptor_stage_flags_, {});
}

pipeline_handler::Write BufferPassBind::GetWrite() const noexcept {
  assert(buffer_);
  return pipeline_handler::Write{
      0,
      descriptor_type_,
      {},
      {vk::DescriptorBufferInfo(buffer_->GetPhysicalBuffer().GetBuffer(), 0,
                                buffer_->GetPhysicalBuffer().GetSize())}};
}

}  // namespace render_graph