#include "raytracer-2.h"

#include <stdint.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "base/base.h"
#include "gpu_resources/image.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "pipeline_handler/descriptor_binding.h"
#include "utill/logger.h"

namespace examples {

RayGenPass::RayGenPass(const CameraInfo* camera_info_source,
                       gpu_resources::Buffer* camera_info,
                       gpu_resources::Buffer* ray_traversal_state,
                       gpu_resources::Buffer* per_pixel_state)
    : camera_info_source_(camera_info_source),
      camera_info_(camera_info),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state) {
  gpu_resources::BufferProperties buffer_properties{};

  buffer_properties.size = sizeof(CameraInfo);
  buffer_properties.memory_flags = {};
  buffer_properties.usage_flags = vk::BufferUsageFlagBits::eUniformBuffer;
  camera_info_->RequireProperties(buffer_properties);

  buffer_properties.memory_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
  buffer_properties.usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
  uint32_t pixel_count_ =
      camera_info_source_->screen_height * camera_info_source_->screen_width;
  buffer_properties.size = pixel_count_ * 4 * 12;
  ray_traversal_state_->RequireProperties(buffer_properties);

  buffer_properties.size = pixel_count_ * 4 * 4;
  per_pixel_state_->RequireProperties(buffer_properties);

  vk::ShaderStageFlags pass_shader_stage = vk::ShaderStageFlagBits::eCompute;
  camera_info_binding_ = pipeline_handler::BufferDescriptorBinding(
      camera_info_, vk::DescriptorType::eUniformBuffer, pass_shader_stage);
  ray_traversal_state_binding_ = pipeline_handler::BufferDescriptorBinding(
      ray_traversal_state_, vk::DescriptorType::eStorageBuffer,
      pass_shader_stage);
  per_pixel_state_binding_ = pipeline_handler::BufferDescriptorBinding(
      per_pixel_state_, vk::DescriptorType::eStorageBuffer, pass_shader_stage);
}

void RayGenPass::OnReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  pipeline_ = pipeline_handler::Compute(
      {
          &camera_info_binding_,
          &ray_traversal_state_binding_,
          &per_pixel_state_binding_,
      },
      pool, {}, "raytrace-raygen.spv", "main");
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

DebugRenderPass::DebugRenderPass(gpu_resources::Image* color_target,
                                 gpu_resources::Buffer* ray_traversal_state,
                                 gpu_resources::Buffer* per_pixel_state)
    : color_target_(color_target),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state) {
  gpu_resources::ImageProperties image_properties;
  image_properties.usage_flags = vk::ImageUsageFlagBits::eStorage;
  image_properties.memory_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
  color_target_->RequireProperties(image_properties);

  gpu_resources::BufferProperties buffer_properties;
  buffer_properties.usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
  buffer_properties.memory_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
  ray_traversal_state_->RequireProperties(buffer_properties);
  per_pixel_state_->RequireProperties(buffer_properties);

  vk::ShaderStageFlags pass_shader_stage = vk::ShaderStageFlagBits::eCompute;
  color_target_binding_ = pipeline_handler::ImageDescriptorBinding(
      color_target_, vk::DescriptorType::eStorageImage, pass_shader_stage,
      vk::ImageLayout::eGeneral);
  ray_traversal_state_binding_ = pipeline_handler::BufferDescriptorBinding(
      ray_traversal_state_, vk::DescriptorType::eStorageBuffer,
      pass_shader_stage);
  per_pixel_state_binding_ = pipeline_handler::BufferDescriptorBinding(
      per_pixel_state_, vk::DescriptorType::eStorageBuffer, pass_shader_stage);
}

void DebugRenderPass::OnReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  pipeline_ = pipeline_handler::Compute(
      {
          &color_target_binding_,
          &ray_traversal_state_binding_,
          &per_pixel_state_binding_,
      },
      pool, {}, "raytrace-debug-render.spv", "main");
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
          sizeof(CameraInfo), vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible});
  gpu_resources::Buffer* ray_traversal_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});
  gpu_resources::Buffer* per_pixel_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});

  raygen_ = RayGenPass(&camera_state_.GetCameraInfo(), camera_info,
                       ray_traversal_state, per_pixel_state);
  render_graph_.AddPass(&raygen_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);
  debug_render_ =
      DebugRenderPass(color_target, ray_traversal_state, per_pixel_state);
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
