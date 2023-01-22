#pragma once

#include "blit_to_swapchain.h"
#include "common.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"

namespace examples {

class RayGenPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;

  const CameraInfo* camera_info_source_ = nullptr;
  gpu_resources::Buffer* camera_info_ = nullptr;
  gpu_resources::Buffer* ray_traversal_state_ = nullptr;
  gpu_resources::Buffer* per_pixel_state_ = nullptr;

  pipeline_handler::BufferDescriptorBinding camera_info_binding_;
  pipeline_handler::BufferDescriptorBinding ray_traversal_state_binding_;
  pipeline_handler::BufferDescriptorBinding per_pixel_state_binding_;

 public:
  RayGenPass() = default;
  RayGenPass(const CameraInfo* camera_info_source,
             gpu_resources::Buffer* camera_info,
             gpu_resources::Buffer* ray_traversal_state,
             gpu_resources::Buffer* per_pixel_state);

  void OnReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

class DebugRenderPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;
  gpu_resources::Image* color_target_ = nullptr;
  gpu_resources::Buffer* ray_traversal_state_ = nullptr;
  gpu_resources::Buffer* per_pixel_state_ = nullptr;

  pipeline_handler::ImageDescriptorBinding color_target_binding_;
  pipeline_handler::BufferDescriptorBinding ray_traversal_state_binding_;
  pipeline_handler::BufferDescriptorBinding per_pixel_state_binding_;

 public:
  DebugRenderPass() = default;
  DebugRenderPass(gpu_resources::Image* color_target,
                  gpu_resources::Buffer* ray_traversal_state,
                  gpu_resources::Buffer* per_pixel_state);

  void OnReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

class RayTracer2 {
  RayGenPass raygen_;
  DebugRenderPass debug_render_;
  BlitToSwapchainPass present_;
  render_graph::RenderGraph render_graph_;
  vk::Semaphore ready_to_present_;
  MainCamera camera_state_;
  void* camera_info_buffer_mapping_ = nullptr;

 public:
  RayTracer2();
  RayTracer2(const RayTracer2&) = delete;
  void operator=(const RayTracer2&) = delete;

  bool Draw();

  ~RayTracer2();
};

}  // namespace examples
