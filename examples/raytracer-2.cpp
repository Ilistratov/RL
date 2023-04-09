#include "raytracer-2.h"

#include <stdint.h>
#include <type_traits>
#include <vector>
#include <vulkan/vulkan.hpp>
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
#include "render_data/bvh.h"
#include "shader/loader.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace examples {

const static char* kRayGenShaderPath = "shaders/raytrace/raygen.spv";
const static char* kTraversePrimaryShaderPath =
    "shaders/raytrace/traverse-primary.spv";
const static char* kDebugRendererShaderPath =
    "shaders/raytrace/debug_render.spv";

TransferToGPUPass::TransferToGPUPass(
    std::vector<TransferRequest> transfer_requests,
    gpu_resources::ResourceManager& resource_manager)
    : transfer_requests_(transfer_requests) {
  DCHECK(!transfer_requests_.empty());
  alingnment_ = base::Base::Get()
                    .GetContext()
                    .GetPhysicalDevice()
                    .getProperties()
                    .limits.nonCoherentAtomSize;
  gpu_resources::BufferProperties transfer_dst_props;
  transfer_dst_props.usage_flags = vk::BufferUsageFlagBits::eTransferDst;
  gpu_resources::BufferProperties transfer_src_props;
  transfer_src_props.usage_flags = vk::BufferUsageFlagBits::eTransferSrc;
  transfer_src_props.size = 0;
  transfer_src_props.allocation_flags =
      VmaAllocationCreateFlagBits::
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT;

  transfer_dst_props.usage_flags = vk::BufferUsageFlagBits::eTransferDst;
  vk::DeviceSize staging_size = 0;
  for (uint32_t idx = 0; idx < transfer_requests_.size(); idx++) {
    DCHECK(transfer_requests_[idx].dst_buffer != nullptr)
        << "Unexpected nullptr in dst_buffer at idx = " << idx;
    DCHECK(transfer_requests_[idx].size > 0)
        << "Unexpected 0 size transfer at idx = " << idx;
    DCHECK(transfer_requests_[idx].data_source != nullptr)
        << "Unexpected nullptr un data_source at idx = " << idx;
    transfer_dst_props.size = transfer_requests_[idx].size;
    transfer_requests_[idx].dst_buffer->RequireProperties(transfer_dst_props);
    staging_size += transfer_requests_[idx].size;
    staging_size = AllignedOffset(staging_size);
  }

  transfer_src_props.size = staging_size;
  staging_buffer_ = resource_manager.AddBuffer(transfer_src_props);
}

TransferToGPUPass::TransferToGPUPass(TransferToGPUPass&& other) noexcept {
  Swap(other);
}

void TransferToGPUPass::operator=(TransferToGPUPass&& other) noexcept {
  TransferToGPUPass tmp(std::move(other));
  Swap(tmp);
}

void TransferToGPUPass::Swap(TransferToGPUPass& other) noexcept {
  Pass::Swap(other);
  transfer_requests_.swap(other.transfer_requests_);
  std::swap(staging_buffer_, other.staging_buffer_);
  std::swap(alingnment_, other.alingnment_);
}

void TransferToGPUPass::OnResourcesInitialized() noexcept {
  vk::DeviceSize dst_offset = 0;
  for (auto& transfer_request : transfer_requests_) {
    dst_offset = staging_buffer_->LoadDataFromPtr(
        transfer_request.data_source, transfer_request.size, dst_offset);
    dst_offset = AllignedOffset(dst_offset);
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
    src_offset = AllignedOffset(src_offset + transfer_request.size);
  }
  transfer_requests_.clear();
}

vk::DeviceSize TransferToGPUPass::AllignedOffset(vk::DeviceSize offset) {
  if (offset % alingnment_ != 0) {
    return offset + alingnment_ - offset % alingnment_;
  }
  return offset;
}

RayGenPass::RayGenPass(const shader::Loader& raygen_shader,
                       pipeline_handler::DescriptorSet* d_set,
                       const CameraInfo* camera_info_source,
                       gpu_resources::Buffer* ray_traversal_state,
                       gpu_resources::Buffer* per_pixel_state)
    : camera_info_pc_range_(raygen_shader.GeneratePushConstantRanges()[0]),
      camera_info_source_(camera_info_source),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state) {
  gpu_resources::BufferProperties buffer_properties{};

  buffer_properties.size = sizeof(CameraInfo);

  uint32_t pixel_count_ =
      camera_info_source_->screen_height * camera_info_source_->screen_width;
  buffer_properties.size = pixel_count_ * 4 * 12;
  ray_traversal_state_->RequireProperties(buffer_properties);

  buffer_properties.size = pixel_count_ * 4 * 4;
  per_pixel_state_->RequireProperties(buffer_properties);

  d_set->BulkBind(std::vector<gpu_resources::Buffer*>{ray_traversal_state_,
                                                      per_pixel_state_});
  pipeline_ = pipeline_handler::Compute(raygen_shader, {d_set});
}

RayGenPass::RayGenPass(RayGenPass&& other) noexcept {
  Swap(other);
}

void RayGenPass::operator=(RayGenPass&& other) noexcept {
  RayGenPass tmp(std::move(other));
  Swap(tmp);
}

void RayGenPass::Swap(RayGenPass& other) noexcept {
  pipeline_.Swap(other.pipeline_);
  std::swap(camera_info_pc_range_, other.camera_info_pc_range_);
  std::swap(camera_info_source_, other.camera_info_source_);
  std::swap(ray_traversal_state_, other.ray_traversal_state_);
  std::swap(per_pixel_state_, other.per_pixel_state_);
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
  primary_cmd.pushConstants(pipeline_.GetLayout(),
                            vk::ShaderStageFlagBits::eCompute,
                            camera_info_pc_range_.offset,
                            camera_info_pc_range_.size, camera_info_source_);
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
      per_pixel_state_(per_pixel_state) {
  gpu_resources::ImageProperties image_properties;
  color_target_->RequireProperties(image_properties);

  gpu_resources::BufferProperties buffer_properties;
  ray_traversal_state_->RequireProperties(buffer_properties);
  per_pixel_state_->RequireProperties(buffer_properties);

  d_set->BulkBind(std::vector<gpu_resources::Buffer*>{ray_traversal_state,
                                                      per_pixel_state_},
                  true);
  d_set->BulkBind({color_target_}, true);
  pipeline_ = pipeline_handler::Compute(debug_render_shader, {d_set});
}

DebugRenderPass::DebugRenderPass(DebugRenderPass&& other) noexcept {
  Swap(other);
}

void DebugRenderPass::operator=(DebugRenderPass&& other) noexcept {
  DebugRenderPass tmp(std::move(other));
  Swap(tmp);
}

void DebugRenderPass::Swap(DebugRenderPass& other) noexcept {
  pipeline_.Swap(other.pipeline_);
  std::swap(color_target_, other.color_target_);
  std::swap(ray_traversal_state_, other.ray_traversal_state_);
  std::swap(per_pixel_state_, other.per_pixel_state_);
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
  required_image_props.usage_flags = vk::ImageUsageFlagBits::eStorage;
  color_target_->RequireProperties(required_image_props);
  depth_target_->RequireProperties(required_image_props);

  pipeline_handler::DescriptorSet* ray_trace_state_dset =
      trace_primary_shader.GenerateDescriptorSet(pool, 1);
  ray_trace_state_dset->BulkBind(
      std::vector<gpu_resources::Buffer*>{ray_traversal_state, per_pixel_state},
      true);

  pipeline_handler::DescriptorSet* image_target_dset =
      trace_primary_shader.GenerateDescriptorSet(pool, 2);
  image_target_dset->BulkBind(
      std::vector<gpu_resources::Image*>{color_target_, depth_target_}, true);

  pipeline_ = pipeline_handler::Compute(
      trace_primary_shader,
      {geometry_buffers_.GetDSet(), ray_trace_state_dset, image_target_dset});
}

TracePrimaryPass::TracePrimaryPass(TracePrimaryPass&& other) noexcept {
  Swap(other);
}

void TracePrimaryPass::operator=(TracePrimaryPass&& other) noexcept {
  TracePrimaryPass tmp(std::move(other));
  Swap(tmp);
}

void TracePrimaryPass::Swap(TracePrimaryPass& other) noexcept {
  pipeline_.Swap(other.pipeline_);
  std::swap(color_target_, other.color_target_);
  std::swap(depth_target_, other.depth_target_);
  std::swap(ray_traversal_state_, other.ray_traversal_state_);
  std::swap(per_pixel_state_, other.per_pixel_state_);
  std::swap(geometry_buffers_, other.geometry_buffers_);
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

RayTracer2::RayTracer2(const std::string& scene_obj_path) {
  scene_mesh_ = render_data::Mesh::LoadFromObj(scene_obj_path);
  scene_bvh_ =
      render_data::BVH(render_data::BVH::BuildPrimitivesBB(scene_mesh_));
  scene_mesh_.ReorderPrimitives(scene_bvh_.GetPrimitiveOrd());

  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  camera_state_ =
      MainCamera(swapchain.GetExtent().width, swapchain.GetExtent().height);
  ready_to_present_ = device.createSemaphore({});
  auto& resource_manager = render_graph_.GetResourceManager();

  gpu_resources::Image* color_target =
      resource_manager.AddImage(gpu_resources::ImageProperties{});
  gpu_resources::Image* depth_target =
      resource_manager.AddImage(gpu_resources::ImageProperties{});
  gpu_resources::Buffer* ray_traversal_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});
  gpu_resources::Buffer* per_pixel_state =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{});

  gpu_resources::Buffer* vertex_pos =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          0, {}, vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer* vertex_ind =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          0, {}, vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer* vertex_nrm =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          scene_mesh_.normal.size() * sizeof(scene_mesh_.normal[0]),
          {},
          vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer* bvh_buffer =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          scene_bvh_.GetNodes().size() * sizeof(scene_bvh_.GetNodes()[0]),
          {},
          vk::BufferUsageFlagBits::eStorageBuffer});
  TransferToGPUPass::TransferRequest pos_transfer{
      vertex_pos, (void*)scene_mesh_.position.data(),
      scene_mesh_.position.size() * sizeof(scene_mesh_.position[0])};
  TransferToGPUPass::TransferRequest ind_transfer{
      vertex_ind, (void*)scene_mesh_.index.data(),
      scene_mesh_.index.size() * sizeof(scene_mesh_.index[0])};
  TransferToGPUPass::TransferRequest nrm_transfer{
      vertex_nrm, (void*)scene_mesh_.normal.data(),
      scene_mesh_.normal.size() * sizeof(scene_mesh_.normal[0])};
  TransferToGPUPass::TransferRequest bvh_transfer{
      bvh_buffer, (void*)scene_bvh_.GetNodes().data(),
      scene_bvh_.GetNodes().size() * sizeof(scene_bvh_.GetNodes()[0])};
  transfer_ = TransferToGPUPass(
      {pos_transfer, ind_transfer, nrm_transfer, bvh_transfer},
      resource_manager);
  render_graph_.AddPass(&transfer_, vk::PipelineStageFlagBits2::eTransfer);
  shader::Loader raygen_shader(kRayGenShaderPath);
  auto raygen_d_set =
      raygen_shader.GenerateDescriptorSet(render_graph_.GetDescriptorPool(), 0);
  raygen_ =
      RayGenPass(raygen_shader, raygen_d_set, &camera_state_.GetCameraInfo(),
                 ray_traversal_state, per_pixel_state);
  render_graph_.AddPass(&raygen_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);

  gpu_resources::Buffer* buf_arr[4] = {vertex_pos, vertex_ind, vertex_nrm,
                                       bvh_buffer};
  BufferBatch<4> geometry_buffers(buf_arr);
  shader::Loader traverse_primary_shader(kTraversePrimaryShaderPath);
  geometry_buffers.SetDSet(traverse_primary_shader.GenerateDescriptorSet(
      render_graph_.GetDescriptorPool(), 0));
  trace_primary_ = TracePrimaryPass(
      traverse_primary_shader, render_graph_.GetDescriptorPool(), color_target,
      depth_target, ray_traversal_state, per_pixel_state, geometry_buffers);
  render_graph_.AddPass(&trace_primary_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);

  present_ = BlitToSwapchainPass(color_target);
  render_graph_.AddPass(&present_, vk::PipelineStageFlagBits2KHR::eTransfer,
                        ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
  camera_state_.SetPos(glm::vec3{0, 250, 0});
}

bool RayTracer2::Draw() {
  camera_state_.Update();
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
