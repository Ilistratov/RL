#include "render_graph/pass.h"

#include "utill/error_handling.h"

namespace render_graph {

void Pass::RecordPostPassParriers(vk::CommandBuffer cmd) {
  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers_;
  for (auto& [name, buffer] : buffer_binds_) {
    vk::BufferMemoryBarrier2KHR barrier = buffer.GetBarrier(user_ind_);
    if (barrier.buffer) {
      buffer_barriers_.push_back(barrier);
    }
  }

  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers_;
  for (auto& [name, image] : image_binds_) {
    vk::ImageMemoryBarrier2KHR barrier = image.GetBarrier(user_ind_);
    if (barrier.image) {
      image_barriers_.push_back(barrier);
    }
  }

  vk::DependencyInfoKHR dep_info({}, {}, buffer_barriers_, image_barriers_);
  cmd.pipelineBarrier2KHR(dep_info);
}

void Pass::OnRecord(vk::CommandBuffer,
                    const std::vector<vk::CommandBuffer>&) noexcept {}

Pass::Pass(uint32_t secondary_cmd_count, vk::PipelineStageFlags2KHR stage_flags)
    : secondary_cmd_count_(secondary_cmd_count), stage_flags_(stage_flags) {}

void Pass::BindResources(uint32_t user_ind,
                         gpu_resources::ResourceManager& resource_manager) {
  DCHECK(user_ind_ == (uint32_t)(-1)) << "Resources already bound";
  user_ind_ = user_ind;
  for (auto& [name, buffer] : buffer_binds_) {
    buffer.OnResourceBind(user_ind_, &resource_manager.GetBuffer(name));
  }
  for (auto& [name, image] : image_binds_) {
    image.OnResourceBind(user_ind_, &resource_manager.GetImage(name));
  }
}

void Pass::ReserveDescriptorSets(pipeline_handler::DescriptorPool&) noexcept {}

void Pass::OnResourcesInitialized() noexcept {}

void Pass::OnWorkloadRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>& secondary_cmd) {
  DCHECK(user_ind_ != uint32_t(-1))
      << "Resources must be bound to use this method";
  OnRecord(primary_cmd, secondary_cmd);
  RecordPostPassParriers(primary_cmd);
}

uint32_t Pass::GetSecondaryCmdCount() const {
  return secondary_cmd_count_;
}
vk::PipelineStageFlags2KHR Pass::GetStageFlags() const {
  return stage_flags_;
}

}  // namespace render_graph
