#pragma once

#include <list>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace gpu_executor {

const uint32_t kCmdPoolMaxAllocStep = 256;

class CommandPool {
  vk::CommandPool cmd_pool_;
  uint32_t primary_alloc_step_ = 1;
  std::vector<vk::CommandBuffer> primary_cmd_;
  uint32_t secondary_alloc_step_ = 1;
  std::vector<vk::CommandBuffer> secondary_cmd_;

  struct InProgressBatch {
    std::vector<vk::CommandBuffer> primary_cmd;
    std::vector<vk::CommandBuffer> secondary_cmd;
    vk::Fence on_submit_finished;
  };
  std::list<InProgressBatch> in_progress_batches_;

  void CheckInprogressBatches();
  std::vector<vk::CommandBuffer>& GetCmdVec(vk::CommandBufferLevel cmd_level);
  uint32_t& GetCmdAllocStep(vk::CommandBufferLevel cmd_level);

 public:
  CommandPool();

  CommandPool(const CommandPool&) = delete;
  void operator=(const CommandPool&) = delete;

  CommandPool(CommandPool&& other) noexcept;
  void operator=(CommandPool&& other) noexcept;
  void Swap(CommandPool& other) noexcept;

  std::vector<vk::CommandBuffer> GetCmd(vk::CommandBufferLevel cmd_level,
                                        uint32_t cmd_count);
  void RecycleCmd(const std::vector<vk::CommandBuffer>& primary_cmd,
                  const std::vector<vk::CommandBuffer>& secondary_cmd,
                  vk::Fence fence);

  ~CommandPool();
};

}  // namespace gpu_executor
