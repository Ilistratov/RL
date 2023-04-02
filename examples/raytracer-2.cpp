#include "raytracer-2.h"

#include <stdint.h>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"
#include "shader/loader.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace examples {

TransferToGPUPass::TransferToGPUPass(
    std::vector<TransferRequest> transfer_requests,
    gpu_resources::ResourceManager& resource_manager)
    : transfer_requests_(transfer_requests) {
  DCHECK(!transfer_requests_.empty());
  gpu_resources::BufferProperties transfer_dst_props;
  transfer_dst_props.usage_flags = vk::BufferUsageFlagBits::eTransferDst;
  gpu_resources::BufferProperties transfer_src_props;
  transfer_src_props.usage_flags = vk::BufferUsageFlagBits::eTransferSrc;
  transfer_src_props.size = 0;
  transfer_dst_props.allocation_flags =
      VmaAllocationCreateFlagBits::
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT;

  transfer_dst_props.usage_flags = vk::BufferUsageFlagBits::eTransferDst;
  for (uint32_t idx = 0; idx < transfer_requests_.size(); idx++) {
    DCHECK(transfer_requests_[idx].dst_buffer != nullptr)
        << "Unexpected nullptr in dst_buffer at idx = " << idx;
    DCHECK(transfer_requests_[idx].size > 0)
        << "Unexpected 0 size transfer at idx = " << idx;
    DCHECK(transfer_requests_[idx].data_source != nullptr)
        << "Unexpected nullptr un data_source at idx = " << idx;
    transfer_dst_props.size = transfer_requests_[idx].size;
    transfer_requests_[idx].dst_buffer->RequireProperties(transfer_dst_props);
    transfer_src_props.size += transfer_requests_[idx].size;
  }

  staging_buffer_ = resource_manager.AddBuffer(transfer_src_props);
}

void TransferToGPUPass::OnResourcesInitialized() noexcept {
  vk::DeviceSize dst_offset = 0;
  for (auto& transfer_request : transfer_requests_) {
    dst_offset = staging_buffer_->LoadDataFromPtr(
        transfer_request.data_source, transfer_request.size, dst_offset);
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.flushMappedMemoryRanges(
      staging_buffer_->GetBuffer()->GetMappedMemoryRange());
}

void TransferToGPUPass::OnPreRecord() {
  if (transfer_requests_.empty()) {
    return;
  }
  gpu_resources::ResourceAccess transfer_dst_access;
  transfer_dst_access.access_flags = vk::AccessFlagBits2::eTransferWrite;
  transfer_dst_access.stage_flags = vk::PipelineStageFlagBits2KHR::eTransfer;
  for (auto& transfer_request : transfer_requests_) {
    transfer_request.dst_buffer->DeclareAccess(transfer_dst_access,
                                               GetPassIdx());
  }
  gpu_resources::ResourceAccess transfer_src_access;
  transfer_src_access.access_flags = vk::AccessFlagBits2KHR::eTransferRead;
  transfer_src_access.stage_flags = vk::PipelineStageFlagBits2KHR::eTransfer;
  staging_buffer_->DeclareAccess(transfer_src_access, GetPassIdx());
}
void TransferToGPUPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  vk::DeviceSize src_offset = 0;
  for (auto& transfer_request : transfer_requests_) {
    gpu_resources::Buffer::RecordCopy(primary_cmd, *staging_buffer_,
                                      *transfer_request.dst_buffer, src_offset,
                                      0, transfer_request.size);
    src_offset += transfer_request.size;
  }
  transfer_requests_.clear();
}

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

TracePrimaryPass::TracePrimaryPass(const shader::Loader& trace_primary_shader,
                                   pipeline_handler::DescriptorPool& pool,
                                   gpu_resources::Image* color_target,
                                   gpu_resources::Image* depth_target,
                                   gpu_resources::Buffer* ray_traversal_state,
                                   gpu_resources::Buffer* per_pixel_state,
                                   BufferBatch<4> geometry_buffers)
    : color_target_(color_target),
      depth_target_(depth_target),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state),
      geometry_buffers_(geometry_buffers) {
  gpu_resources::ImageProperties required_image_props;
  required_image_props.format = vk::Format::eR32Sfloat;
  depth_target_->RequireProperties(required_image_props);

  pipeline_handler::DescriptorSet* ray_trace_state_dset =
      trace_primary_shader.GenerateDescriptorSet(pool, 1);
  ray_trace_state_dset->BulkBind(
      std::vector<gpu_resources::Buffer*>{ray_traversal_state, per_pixel_state},
      true);

  pipeline_handler::DescriptorSet* image_target_dset =
      trace_primary_shader.GenerateDescriptorSet(pool, 1);
  image_target_dset->BulkBind(
      std::vector<gpu_resources::Image*>{color_target_, depth_target_}, true);

  pipeline_ = pipeline_handler::Compute(
      trace_primary_shader,
      {geometry_buffers_.GetDSet(), ray_trace_state_dset, image_target_dset});
}

void TracePrimaryPass::OnPreRecord() {
  gpu_resources::ResourceAccess image_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite, vk::ImageLayout::eGeneral};
  color_target_->DeclareAccess(image_access, GetPassIdx());
  depth_target_->DeclareAccess(image_access, GetPassIdx());

  gpu_resources::ResourceAccess static_data_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageRead};
  geometry_buffers_.DeclareCommonAccess(static_data_access, GetPassIdx());

  gpu_resources::ResourceAccess ray_trace_state_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageRead |
          vk::AccessFlagBits2KHR::eShaderStorageWrite};
  ray_traversal_state_->DeclareAccess(ray_trace_state_access, GetPassIdx());
  per_pixel_state_->DeclareAccess(ray_trace_state_access, GetPassIdx());
}

void TracePrimaryPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  vk::Extent2D extent = base::Base::Get().GetSwapchain().GetExtent();
  uint32_t ray_count = extent.width * extent.height;
  const static uint32_t kRaysPerGroup = 16;
  pipeline_.RecordDispatch(primary_cmd, ray_count / kRaysPerGroup, 1, 1);
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
