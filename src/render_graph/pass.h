#pragma once

#include <map>
#include <string>

#include "gpu_executer/task.h"
#include "pipeline_handler/descriptor_pool.h"
#include "render_graph/buffer_pass_bind.h"
#include "render_graph/image_pass_bind.h"
#include "render_graph/resource_manager.h"

namespace render_graph {

class Pass : public gpu_executer::Task {
  uint32_t user_ind_ = -1;
  uint32_t secondary_cmd_count_ = 0;
  vk::PipelineStageFlagBits2KHR stage_flags_ = {};

 protected:
  std::map<std::string, BufferPassBind> buffer_binds_;
  std::map<std::string, ImagePassBind> image_binds_;

 public:
  Pass(uint32_t secondary_cmd_count = 0,
       vk::PipelineStageFlagBits2KHR stage_flags = {});

  void BindResources(uint32_t user_ind, ResourceManager& resource_manager);
  virtual void ReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept;
  virtual void OnResourcesInitialized() noexcept;

  void RecordPostPassParriers(vk::CommandBuffer cmd);
  virtual void OnRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) noexcept;
  void OnWorkloadRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) override;

  uint32_t GetSecondaryCmdCount() const;
  vk::PipelineStageFlagBits2KHR GetStageFlags() const;
};

}  // namespace render_graph
