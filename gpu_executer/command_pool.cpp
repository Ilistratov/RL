#include "gpu_executer/command_pool.h"

#include "base/base.h"

namespace gpu_executer {

void CommandPool::CheckInprogressBatches() {
  auto device = base::Base::Get().GetContext().GetDevice();
  auto it = in_progress_batches_.begin();
  while (it != in_progress_batches_.end()) {
    if (device.getFenceStatus(it->on_submit_finished) == vk::Result::eSuccess) {
      primary_cmd_.insert(primary_cmd_.begin(), it->primary_cmd.begin(),
                          it->primary_cmd.end());
      secondary_cmd_.insert(secondary_cmd_.begin(), it->secondary_cmd.begin(),
                            it->secondary_cmd.end());
      it = in_progress_batches_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<vk::CommandBuffer>& CommandPool::GetCmdVec(
    vk::CommandBufferLevel cmd_level) {
  if (cmd_level == vk::CommandBufferLevel::ePrimary) {
    return primary_cmd_;
  } else {
    return secondary_cmd_;
  }
}
uint32_t& CommandPool::GetCmdAllocStep(vk::CommandBufferLevel cmd_level) {
  if (cmd_level == vk::CommandBufferLevel::ePrimary) {
    return primary_alloc_step_;
  } else {
    return secondary_alloc_step_;
  }
}

std::vector<vk::CommandBuffer> CommandPool::GetCmd(
    vk::CommandBufferLevel cmd_level,
    uint32_t cmd_count) {
  CheckInprogressBatches();
  std::vector<vk::CommandBuffer>& cmd_vec = GetCmdVec(cmd_level);
  if (cmd_vec.size() < cmd_count) {
    uint32_t& alloc_step = GetCmdAllocStep(cmd_level);
    if (alloc_step + cmd_vec.size() < cmd_count) {
      alloc_step = cmd_count - cmd_vec.size();
    }
    assert(alloc_step < kCmdPoolMaxAllocStep);

    auto device = base::Base::Get().GetContext().GetDevice();
    auto n_cmd = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(cmd_pool_, cmd_level, alloc_step));
    cmd_vec.insert(cmd_vec.end(), n_cmd.begin(), n_cmd.end());
    alloc_step *= 2;
  }

  auto it = cmd_vec.begin() + (cmd_vec.size() - cmd_count);
  std::vector<vk::CommandBuffer> res(it, cmd_vec.end());
  cmd_vec.resize(it - cmd_vec.begin());
  return res;
}

void CommandPool::RecycleCmd(
    const std::vector<vk::CommandBuffer>& primary_cmd,
    const std::vector<vk::CommandBuffer>& secondary_cmd,
    vk::Fence on_submit_finished) {
  in_progress_batches_.push_back(
      InProgressBatch{primary_cmd, secondary_cmd, on_submit_finished});
}

}  // namespace gpu_executer
