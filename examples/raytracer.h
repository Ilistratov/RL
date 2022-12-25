#pragma once

#include <vcruntime.h>
#include <vector>

#include "blit_to_swapchain.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "render_graph/layout_initializer_pass.h"
#include "render_graph/render_graph.h"
#include "utill/transform.h"

namespace examples {

struct CameraInfo {
  utill::Transform camera_to_world;
  uint32_t screen_width;
  uint32_t screen_height;
  float aspect;
};

struct GeometryBuffers {
  gpu_resources::Buffer* position;
  gpu_resources::Buffer* normal;
  gpu_resources::Buffer* tex_coord;
  gpu_resources::Buffer* index;
  gpu_resources::Buffer* light;
  gpu_resources::Buffer* bvh;

  size_t AddBuffersToRenderGraph(
      gpu_resources::ResourceManager& resource_manager);

  void AddCommonRequierment(gpu_resources::BufferProperties requierment);
  void DeclareCommonAccess(gpu_resources::ResourceAccess access,
                           uint32_t pass_idx);
};

struct GeometryBindings {
  pipeline_handler::BufferDescriptorBinding position;
  pipeline_handler::BufferDescriptorBinding normal;
  pipeline_handler::BufferDescriptorBinding tex_coord;
  pipeline_handler::BufferDescriptorBinding index;
  pipeline_handler::BufferDescriptorBinding light;
  pipeline_handler::BufferDescriptorBinding bvh;

  GeometryBindings() = default;
  GeometryBindings(GeometryBuffers buffers, vk::ShaderStageFlags access_stage);
};

class ResourceTransferPass : public render_graph::Pass {
  GeometryBuffers geometry_;
  gpu_resources::Buffer* staging_buffer_;
  gpu_resources::Buffer* camera_info_;
  bool is_first_record_ = true;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

 public:
  ResourceTransferPass() = default;
  ResourceTransferPass(GeometryBuffers geometry,
                       gpu_resources::Buffer* staging_buffer,
                       gpu_resources::Buffer* camera_info);

  void OnResourcesInitialized() noexcept override;
};

class RaytracerPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;
  GeometryBuffers geometry_;
  gpu_resources::Image* color_target_;
  gpu_resources::Image* depth_target_;
  gpu_resources::Buffer* camera_info_;

  GeometryBindings geometry_bindings_;
  pipeline_handler::ImageDescriptorBinding color_target_binding_;
  pipeline_handler::ImageDescriptorBinding depth_target_binding_;
  pipeline_handler::BufferDescriptorBinding camera_info_binding_;

 public:
  RaytracerPass() = default;
  RaytracerPass(GeometryBuffers geometry,
                gpu_resources::Image* color_target,
                gpu_resources::Image* depth_target,
                gpu_resources::Buffer* camera_info);

  void OnReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

class RayTracer {
  render_graph::LayoutInitializerPass initializer_;
  ResourceTransferPass resource_transfer_;
  RaytracerPass raytrace_;
  BlitToSwapchainPass present_;
  render_graph::RenderGraph render_graph_;
  vk::Semaphore ready_to_present_;

  GeometryBuffers geometry_;
  gpu_resources::Image* color_target_;
  gpu_resources::Image* depth_target_;
  gpu_resources::Buffer* camera_info_;
  gpu_resources::Buffer* staging_buffer_;

 public:
  RayTracer();
  RayTracer(const RayTracer&) = delete;
  void operator=(const RayTracer&) = delete;

  bool Draw();

  ~RayTracer();
};

}  // namespace examples
