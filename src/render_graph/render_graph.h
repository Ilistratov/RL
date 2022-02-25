#pragma once

#include <memory>
#include <vector>

#include "gpu_executer/executer.h"
#include "pipeline_handler/descriptor_pool.h"
#include "render_graph/pass.h"
#include "render_graph/resource_manager.h"

namespace render_graph {

class RenderGraph {
  ResourceManager resource_manager_;
  gpu_executer::Executer executer_;
  pipeline_handler::DescriptorPool descriptor_pool_;
  std::vector<std::unique_ptr<Pass>> passes_;

 public:
  RenderGraph() = default;

  RenderGraph(const RenderGraph&) = delete;
  void operator=(const RenderGraph&) = delete;

  void AddPass(std::unique_ptr<Pass> pass,
               vk::Semaphore external_signal = {},
               vk::Semaphore external_wait = {});
  ResourceManager& GetResourceManager();
  void Init();
  void RenderFrame();
};

}  // namespace render_graph
