#pragma once

#include <memory>
#include <vector>

#include "gpu_executer/executer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/descriptor_pool.h"
#include "render_graph/pass.h"

namespace render_graph {

class PreFrameResourceInitializerTask : public gpu_executer::Task {
  gpu_resources::PassAccessSyncronizer* access_syncronizer_ = nullptr;
  uint32_t pass_count_ = 0;

 public:
  PreFrameResourceInitializerTask() = default;
  PreFrameResourceInitializerTask(
      gpu_resources::PassAccessSyncronizer* access_syncronizer,
      uint32_t pass_count);

  void OnWorkloadRecord(vk::CommandBuffer cmd,
                        const std::vector<vk::CommandBuffer>&) override;
};

class RenderGraph {
  gpu_resources::ResourceManager resource_manager_;
  gpu_executer::Executer executer_;
  pipeline_handler::DescriptorPool descriptor_pool_;
  PreFrameResourceInitializerTask initialize_task_;
  std::vector<Pass*> passes_;

 public:
  RenderGraph();

  RenderGraph(const RenderGraph&) = delete;
  void operator=(const RenderGraph&) = delete;

  // semaphore wait/signal operations will affect/happen at
  // pipeline stages specified by stage_flags/
  void AddPass(Pass* pass,
               vk::PipelineStageFlags2KHR stage_flags = {},
               vk::Semaphore external_signal = {},
               vk::Semaphore external_wait = {});
  gpu_resources::ResourceManager& GetResourceManager();
  void Init();
  void RenderFrame();
};

}  // namespace render_graph
