#pragma once

#include <vector>

#include "pipeline_handler/compute.h"
#include "render_graph/render_graph.h"
#include "swapchain_present_pass.h"
#include "utill/transform.h"

namespace examples {

struct CameraInfo {
  utill::Transform camera_to_world;
  uint32_t screen_width;
  uint32_t screen_height;
  float aspect;
};

class ResourceTransferPass : public render_graph::Pass {
  bool is_first_record_ = true;

  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

 public:
  ResourceTransferPass();

  void OnResourcesInitialized() noexcept override;
};

class RaytracerPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;
  std::vector<const pipeline_handler::DescriptorBinding*> shader_bindings_;

  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

 public:
  RaytracerPass();

  void ReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;
  void OnResourcesInitialized() noexcept override;
};

class RayTracer {
  ResourceTransferPass resource_transfer_;
  RaytracerPass raytrace_;
  SwapchainPresentPass present_;
  render_graph::RenderGraph render_graph_;
  vk::Semaphore ready_to_present_;

 public:
  RayTracer();
  RayTracer(const RayTracer&) = delete;
  void operator=(const RayTracer&) = delete;

  bool Draw();

  ~RayTracer();
};

}  // namespace examples
