#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "blit_to_swapchain.h"
#include "common.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_pool.h"
#include "pipeline_handler/descriptor_set.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"
#include "shader/loader.h"
#include "utill/error_handling.h"

namespace examples {

class TransferToGPUPass : public render_graph::Pass {
 public:
  struct TransferRequest {
    gpu_resources::Buffer* dst_buffer = nullptr;
    void* data_source = nullptr;
    vk::DeviceSize size = 0;
  };

  TransferToGPUPass(std::vector<TransferRequest> transfer_requests,
                    gpu_resources::ResourceManager& resource_manager);

 private:
  void OnResourcesInitialized() noexcept override;
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

  std::vector<TransferRequest> transfer_requests_;
  gpu_resources::Buffer* staging_buffer_;
};

class RayGenPass : public render_graph::Pass {
 public:
  RayGenPass() = default;
  RayGenPass(const shader::Loader& raygen_shader,
             pipeline_handler::DescriptorSet* d_set,
             const CameraInfo* camera_info_source,
             gpu_resources::Buffer* camera_info,
             gpu_resources::Buffer* ray_traversal_state,
             gpu_resources::Buffer* per_pixel_state);

 private:
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

  pipeline_handler::Compute pipeline_;

  const CameraInfo* camera_info_source_ = nullptr;
  gpu_resources::Buffer* camera_info_ = nullptr;
  gpu_resources::Buffer* ray_traversal_state_ = nullptr;
  gpu_resources::Buffer* per_pixel_state_ = nullptr;

  pipeline_handler::DescriptorSet* d_set_ = nullptr;
};

class DebugRenderPass : public render_graph::Pass {
 public:
  DebugRenderPass() = default;
  DebugRenderPass(const shader::Loader& debug_render_shader,
                  pipeline_handler::DescriptorSet* d_set,
                  gpu_resources::Image* color_target,
                  gpu_resources::Buffer* ray_traversal_state,
                  gpu_resources::Buffer* per_pixel_state);

 private:
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

  pipeline_handler::Compute pipeline_;
  gpu_resources::Image* color_target_ = nullptr;
  gpu_resources::Buffer* ray_traversal_state_ = nullptr;
  gpu_resources::Buffer* per_pixel_state_ = nullptr;
  pipeline_handler::DescriptorSet* d_set_ = nullptr;
};

template <uint32_t N>
class BufferBatch {
  gpu_resources::Buffer* buffers_[N];
  pipeline_handler::DescriptorSet* dset_ = nullptr;

 public:
  BufferBatch() = default;
  BufferBatch(gpu_resources::Buffer* buffers[N]) {
    for (uint32_t idx = 0; idx < N; idx++) {
      DCHECK(buffers[idx] != nullptr);
      buffers_[idx] = buffers[idx];
    }
  }

  void RequireCommonProps(gpu_resources::BufferProperties props) {
    for (uint32_t idx = 0; idx < N; idx++) {
      buffers_[idx]->RequireProperties(props);
    }
  }

  void DeclareCommonAccess(gpu_resources::ResourceAccess access,
                           uint32_t pass_idx) {
    for (uint32_t idx = 0; idx < N; idx++) {
      buffers_[idx]->DeclareAccess(access, pass_idx);
    }
  }

  pipeline_handler::DescriptorSet* GetDSet() const { return dset_; }

  void SetDSet(pipeline_handler::DescriptorSet* dset) {
    DCHECK(dset != nullptr);
    DCHECK(dset_ == nullptr) << "Descriptor set is already set";
    dset_ = dset;
    for (uint32_t idx = 0; idx < N; idx++) {
      dset_->GetBufferBinding(idx)->SetBuffer(buffers_[idx], true);
    }
  }
};

class TracePrimaryPass : public render_graph::Pass {
 public:
  TracePrimaryPass(const shader::Loader& trace_primary_shader,
                   pipeline_handler::DescriptorPool& pool,
                   gpu_resources::Image* color_target,
                   gpu_resources::Image* depth_target,
                   gpu_resources::Buffer* ray_traversal_state,
                   gpu_resources::Buffer* per_pixel_state,
                   BufferBatch<4> geometry_buffers);

 private:
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

  pipeline_handler::Compute pipeline_;
  gpu_resources::Image* color_target_ = nullptr;
  gpu_resources::Image* depth_target_ = nullptr;
  gpu_resources::Buffer* ray_traversal_state_ = nullptr;
  gpu_resources::Buffer* per_pixel_state_ = nullptr;
  BufferBatch<4> geometry_buffers_;
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
