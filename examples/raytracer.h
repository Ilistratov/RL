#pragma once

#include <glm/fwd.hpp>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "blit_to_swapchain.h"
#include "common.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"
#include "render_data/bvh.h"
#include "render_data/mesh.h"
#include "render_graph/render_graph.h"
#include "shader/loader.h"

namespace examples {

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

  pipeline_handler::DescriptorSet* d_set_;

 public:
  RaytracerPass() = default;
  RaytracerPass(const shader::Loader& raytrace_shader,
                pipeline_handler::DescriptorSet* d_set,
                GeometryBuffers geometry,
                gpu_resources::Image* color_target,
                gpu_resources::Image* depth_target,
                gpu_resources::Buffer* camera_info);

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
};

class RayTracer {
  ResourceTransferPass resource_transfer_;
  RaytracerPass raytrace_;
  BlitToSwapchainPass present_;
  render_graph::RenderGraph render_graph_;
  vk::Semaphore ready_to_present_;

 public:
  RayTracer(render_data::Mesh& mesh, render_data::BVH const& bvh);
  RayTracer(const RayTracer&) = delete;
  void operator=(const RayTracer&) = delete;

  bool Draw();
  void SetCameraPosition(glm::vec3 pos);

  ~RayTracer();
};

}  // namespace examples
