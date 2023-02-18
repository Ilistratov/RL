#include "raytracer-2.h"

#include <stdint.h>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "shader/loader.h"
#include "utill/logger.h"

namespace examples {

RayGenPass::RayGenPass(const shader::Loader& raygen_shader,
                       pipeline_handler::DescriptorSet* d_set,
                       const CameraInfo* camera_info_source,
                       gpu_resources::Buffer* camera_info,
                       gpu_resources::Buffer* ray_traversal_state,
                       gpu_resources::Buffer* per_pixel_state)
    : camera_info_source_(camera_info_source),
      camera_info_(camera_info),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state),
      d_set_(d_set) {
  gpu_resources::BufferProperties buffer_properties{};

  buffer_properties.size = sizeof(CameraInfo);
  camera_info_->RequireProperties(buffer_properties);

  uint32_t pixel_count_ =
      camera_info_source_->screen_height * camera_info_source_->screen_width;
  buffer_properties.size = pixel_count_ * 4 * 12;
  ray_traversal_state_->RequireProperties(buffer_properties);

  buffer_properties.size = pixel_count_ * 4 * 4;
  per_pixel_state_->RequireProperties(buffer_properties);

  d_set_->BulkBind({camera_info_, ray_traversal_state_, per_pixel_state_});
  pipeline_ = pipeline_handler::Compute(raygen_shader, {d_set_});
}

void RayGenPass::OnPreRecord() {
  gpu_resources::ResourceAccess buffer_access{};
  buffer_access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageWrite;
  buffer_access.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  ray_traversal_state_->DeclareAccess(buffer_access, GetPassIdx());
  per_pixel_state_->DeclareAccess(buffer_access, GetPassIdx());
}

void RayGenPass::OnRecord(vk::CommandBuffer primary_cmd,
                          const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  pipeline_.RecordDispatch(primary_cmd, swapchain.GetExtent().width / 8,
                           swapchain.GetExtent().height / 8, 1);
}

DebugRenderPass::DebugRenderPass(const shader::Loader& debug_render_shader,
                                 pipeline_handler::DescriptorSet* d_set,
                                 gpu_resources::Image* color_target,
                                 gpu_resources::Buffer* ray_traversal_state,
                                 gpu_resources::Buffer* per_pixel_state)
    : color_target_(color_target),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state),
      d_set_(d_set) {
  gpu_resources::ImageProperties image_properties;
  color_target_->RequireProperties(image_properties);

  gpu_resources::BufferProperties buffer_properties;
  ray_traversal_state_->RequireProperties(buffer_properties);
  per_pixel_state_->RequireProperties(buffer_properties);

  d_set_->BulkBind(std::vector<gpu_resources::Buffer*>{ray_traversal_state,
                                                       per_pixel_state_},
                   true);
  d_set_->BulkBind({color_target_}, true);
  pipeline_ = pipeline_handler::Compute(debug_render_shader, {d_set});
}

void DebugRenderPass::OnPreRecord() {
  gpu_resources::ResourceAccess access{};
  access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageRead;
  access.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  ray_traversal_state_->DeclareAccess(access, GetPassIdx());
  per_pixel_state_->DeclareAccess(access, GetPassIdx());

  access.layout = vk::ImageLayout::eGeneral;
  access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageWrite;
  color_target_->DeclareAccess(access, GetPassIdx());
}

void DebugRenderPass::OnRecord(vk::CommandBuffer primary_cmd,
                               const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  pipeline_.RecordDispatch(
      primary_cmd,
      (swapchain.GetExtent().width * swapchain.GetExtent().height) / 64, 1, 1);
}

RayTracer2::RayTracer2() {
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  camera_state_ =
      MainCamera(swapchain.GetExtent().width, swapchain.GetExtent().height);
  ready_to_present_ = device.createSemaphore({});
  auto& resource_manager = render_graph_.GetResourceManager();

  gpu_resources::Image* color_target =
      resource_manager.AddImage(gpu_resources::ImageProperties{});
  gpu_resources::Buffer* camera_info =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          sizeof(CameraInfo),
          VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT |
              VmaAllocationCreateFlagBits::
                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          vk::BufferUsageFlagBits::eUniformBuffer});
  gpu_resources::Buffer* ray_traversal_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});
  gpu_resources::Buffer* per_pixel_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});

  shader::Loader raygen_shader("shader/raytrace/raygen.spv");
  auto raygen_d_set =
      raygen_shader.GenerateDescriptorSet(render_graph_.GetDescriptorPool(), 0);
  raygen_ =
      RayGenPass(raygen_shader, raygen_d_set, &camera_state_.GetCameraInfo(),
                 camera_info, ray_traversal_state, per_pixel_state);
  render_graph_.AddPass(&raygen_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);
  shader::Loader debug_render_shader("shader/raytrace/debug_render.spv");
  auto debug_render_d_set = debug_render_shader.GenerateDescriptorSet(
      render_graph_.GetDescriptorPool(), 0);
  debug_render_ =
      DebugRenderPass(debug_render_shader, debug_render_d_set, color_target,
                      ray_traversal_state, per_pixel_state);
  render_graph_.AddPass(&debug_render_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);
  present_ = BlitToSwapchainPass(color_target);
  render_graph_.AddPass(&present_, vk::PipelineStageFlagBits2KHR::eTransfer,
                        ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();

  camera_info_buffer_mapping_ = camera_info->GetBuffer()->GetMappingStart();
}

bool RayTracer2::Draw() {
  camera_state_.Update();
  memcpy(camera_info_buffer_mapping_, &camera_state_.GetCameraInfo(),
         sizeof(CameraInfo));
  auto& swapchain = base::Base::Get().GetSwapchain();

  if (!swapchain.AcquireNextImage()) {
    LOG << "Failed to acquire next image";
    return false;
  }
  swapchain.GetActiveImageInd();
  render_graph_.RenderFrame();
  if (swapchain.Present(ready_to_present_) != vk::Result::eSuccess) {
    LOG << "Failed to present";
    return false;
  }
  return true;
}

RayTracer2::~RayTracer2() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.waitIdle();
  device.destroySemaphore(ready_to_present_);
}

}  // namespace examples
