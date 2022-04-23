#pragma once

#include "pipeline_handler/compute.h"
#include "render_graph/render_graph.h"

namespace examples {

class BlitToSwapchainPass : public render_graph::Pass {
  const std::string& render_target_name_;

 public:
  BlitToSwapchainPass(const std::string& render_target_name);
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

}