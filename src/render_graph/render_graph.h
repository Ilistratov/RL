#pragma once

#include <memory>
#include <vector>

#include "gpu_executer/executer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/descriptor_pool.h"
#include "render_graph/pass.h"

namespace render_graph {

class RenderGraph {
  gpu_resources::ResourceManager resource_manager_;
  gpu_executer::Executer executer_;
  pipeline_handler::DescriptorPool descriptor_pool_;
  std::vector<Pass*> passes_;

 public:
  RenderGraph() = default;

  RenderGraph(const RenderGraph&) = delete;
  void operator=(const RenderGraph&) = delete;

  void AddPass(Pass* pass,
               vk::Semaphore external_signal = {},
               vk::Semaphore external_wait = {});
  gpu_resources::ResourceManager& GetResourceManager();
  void Init();
  void RenderFrame();
};

}  // namespace render_graph
