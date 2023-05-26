#include "render_graph/pass.h"

#include <stdint.h>
#include <type_traits>
#include "utill/error_handling.h"

namespace render_graph {

void Pass::RecordPostPassParriers(vk::CommandBuffer cmd) {
  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers_ =
      access_syncronizer_->GetBufferPostPassBarriers(pass_idx_);
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers_ =
      access_syncronizer_->GetImagePostPassBarriers(pass_idx_);

  vk::DependencyInfoKHR dep_info({}, {}, buffer_barriers_, image_barriers_);
  cmd.pipelineBarrier2KHR(dep_info);
}

Pass::Pass(Pass&& other) noexcept {
  Swap(other);
}

void Pass::operator=(Pass&& other) noexcept {
  Pass tmp(std::move(other));
  Swap(tmp);
}

void Pass::Swap(Pass& other) noexcept {
  DCHECK(pass_idx_ == uint32_t(-1)) << "Bound passes can not be moved";
  DCHECK(other.pass_idx_ == uint32_t(-1)) << "Bound passes can not be moved";
  std::swap(access_syncronizer_, other.access_syncronizer_);
  std::swap(pass_idx_, other.pass_idx_);
  std::swap(secondary_cmd_count_, other.secondary_cmd_count_);
}

void Pass::OnPreRecord() {}

void Pass::OnRecord(vk::CommandBuffer, const std::vector<vk::CommandBuffer>&) {}

Pass::Pass(uint32_t secondary_cmd_count)
    : pass_idx_(-1), secondary_cmd_count_(secondary_cmd_count) {}

void Pass::OnRegister(
    uint32_t pass_idx,
    gpu_resources::PassAccessSyncronizer* access_syncronizer) {
  DCHECK(access_syncronizer != nullptr)
      << "access_syncronizer must be valid PassAccessSyncronizer";
  pass_idx_ = pass_idx;
  access_syncronizer_ = access_syncronizer;
}

void Pass::OnResourcesInitialized() noexcept {}

void Pass::OnWorkloadRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>& secondary_cmd) {
  DCHECK(pass_idx_ != (uint32_t)-1) << "Pass was not registered";
  OnRecord(primary_cmd, secondary_cmd);
  RecordPostPassParriers(primary_cmd);
}

uint32_t Pass::GetPassIdx() const {
  return pass_idx_;
}

uint32_t Pass::GetSecondaryCmdCount() const {
  return secondary_cmd_count_;
}

}  // namespace render_graph
