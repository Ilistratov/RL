#pragma once

#include "pipeline_handler/compute.h"
#include "render_graph/render_graph.h"
#include "swapchain_present_pass.h"

namespace examples {

struct PushConstants {
  uint32_t s_width = 0;
  uint32_t s_height = 0;
  float center_x = 0;
  float center_y = 0;
  float scale = 1.0;
};

class MandelbrotDrawPass : public render_graph::Pass {
  pipeline_handler::Compute compute_pipeline_;
  PushConstants push_constants_;

  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

 public:
  MandelbrotDrawPass();

  void ReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;
  void OnResourcesInitialized() noexcept override;

  PushConstants& GetPushConstants();
};

class Mandelbrot {
  MandelbrotDrawPass draw_;
  SwapchainPresentPass present_;
  vk::Semaphore ready_to_present_;
  render_graph::RenderGraph render_graph_;
  float dst_x_ = 0;
  float dst_y_ = 0;
  float dst_scale_ = 1;

  void UpdatePushConstants();

 public:
  Mandelbrot();
  Mandelbrot(const Mandelbrot&) = delete;
  void operator=(const Mandelbrot&) = delete;

  bool Draw();

  ~Mandelbrot();
};

}  // namespace examples
