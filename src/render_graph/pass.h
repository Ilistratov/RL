#pragma once

#include <map>
#include <string>

#include "gpu_executor/task.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/descriptor_pool.h"

namespace render_graph {

class Pass : public gpu_executor::Task {
  gpu_resources::PassAccessSyncronizer* access_syncronizer_ = nullptr;
  uint32_t pass_idx_ = uint32_t(-1);
  uint32_t secondary_cmd_count_ = uint32_t(0);

 protected:
  virtual void OnReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept;
  virtual void OnRecord(vk::CommandBuffer primary_cmd,
                        const std::vector<vk::CommandBuffer>& secondary_cmd);

  void RecordPostPassParriers(vk::CommandBuffer cmd);

  Pass(Pass&& other) noexcept;
  void operator=(Pass&& other) noexcept;
  void Swap(Pass& other) noexcept;

 public:
  Pass(uint32_t secondary_cmd_count = 0);
  Pass(Pass const&) = delete;
  void operator=(Pass const&) = delete;

  void OnRegister(uint32_t pass_idx,
                  gpu_resources::PassAccessSyncronizer* access_syncronizer,
                  pipeline_handler::DescriptorPool& pool);
  virtual void OnResourcesInitialized() noexcept;
  virtual void OnPreRecord();

  void OnWorkloadRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) override;

  uint32_t GetPassIdx() const;
  uint32_t GetSecondaryCmdCount() const;
};

}  // namespace render_graph
