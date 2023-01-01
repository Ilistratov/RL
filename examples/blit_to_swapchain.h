#pragma once

#include "gpu_resources/image.h"
#include "pipeline_handler/compute.h"
#include "render_graph/render_graph.h"

namespace examples {

class BlitToSwapchainPass : public render_graph::Pass {
  gpu_resources::Image* render_target_;

 public:
  BlitToSwapchainPass() = default;
  BlitToSwapchainPass(gpu_resources::Image* render_target);
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

}  // namespace examples
