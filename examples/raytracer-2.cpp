#include "raytracer-2.h"

#include <array>
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
#include "render_data/mesh.h"
#include "shader/loader.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace examples {

const static char *kRayGenShaderPath = "shaders/raytrace/raygen.spv";
const static char *kTraversePrimaryShaderPath =
    "shaders/raytrace/traverse-primary.spv";
const static char *kTraverseShadowShaderPath =
    "shaders/raytrace/traverse-shadow.spv";
// const static char *kDebugRendererShaderPath =
//     "shaders/raytrace/debug_render.spv";

const static uint32_t kTraversalStateSize = 48;
const static uint32_t kPerPixelStateSize = 16;

TransferToGPUPass::TransferToGPUPass(
    std::vector<TransferRequest> transfer_requests,
    gpu_resources::ResourceManager &resource_manager)
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

TransferToGPUPass::TransferToGPUPass(TransferToGPUPass &&other) noexcept {
  Swap(other);
}

void TransferToGPUPass::operator=(TransferToGPUPass &&other) noexcept {
  TransferToGPUPass tmp(std::move(other));
  Swap(tmp);
}

void TransferToGPUPass::Swap(TransferToGPUPass &other) noexcept {
  Pass::Swap(other);
  transfer_requests_.swap(other.transfer_requests_);
  std::swap(staging_buffer_, other.staging_buffer_);
  std::swap(alingnment_, other.alingnment_);
}

void TransferToGPUPass::OnResourcesInitialized() noexcept {
  vk::DeviceSize dst_offset = 0;
  for (auto &transfer_request : transfer_requests_) {
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
  for (auto &transfer_request : transfer_requests_) {
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
    const std::vector<vk::CommandBuffer> &) noexcept {
  vk::DeviceSize src_offset = 0;
  for (auto &transfer_request : transfer_requests_) {
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

RayGenPass::RayGenPass(const shader::Loader &raygen_shader,
                       pipeline_handler::DescriptorSet *d_set,
                       const CameraInfo *camera_info_source,
                       gpu_resources::Buffer *ray_traversal_state,
                       gpu_resources::Buffer *per_pixel_state)
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

  d_set->BulkBind(std::vector<gpu_resources::Buffer *>{ray_traversal_state_,
                                                       per_pixel_state_});
  pipeline_ = pipeline_handler::Compute(raygen_shader, {d_set});
}

RayGenPass::RayGenPass(RayGenPass &&other) noexcept { Swap(other); }

void RayGenPass::operator=(RayGenPass &&other) noexcept {
  RayGenPass tmp(std::move(other));
  Swap(tmp);
}

void RayGenPass::Swap(RayGenPass &other) noexcept {
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
                          const std::vector<vk::CommandBuffer> &) noexcept {
  auto &swapchain = base::Base::Get().GetSwapchain();
  primary_cmd.pushConstants(pipeline_.GetLayout(),
                            vk::ShaderStageFlagBits::eCompute,
                            camera_info_pc_range_.offset,
                            camera_info_pc_range_.size, camera_info_source_);
  pipeline_.RecordDispatch(primary_cmd, swapchain.GetExtent().width / 8,
                           swapchain.GetExtent().height / 8, 1);
}

DebugRenderPass::DebugRenderPass(const shader::Loader &debug_render_shader,
                                 pipeline_handler::DescriptorSet *d_set,
                                 gpu_resources::Image *color_target,
                                 gpu_resources::Buffer *ray_traversal_state,
                                 gpu_resources::Buffer *per_pixel_state)
    : color_target_(color_target), ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state) {
  gpu_resources::ImageProperties image_properties;
  color_target_->RequireProperties(image_properties);

  gpu_resources::BufferProperties buffer_properties;
  ray_traversal_state_->RequireProperties(buffer_properties);
  per_pixel_state_->RequireProperties(buffer_properties);

  d_set->BulkBind(std::vector<gpu_resources::Buffer *>{ray_traversal_state,
                                                       per_pixel_state_},
                  true);
  d_set->BulkBind({color_target_}, true);
  pipeline_ = pipeline_handler::Compute(debug_render_shader, {d_set});
}

DebugRenderPass::DebugRenderPass(DebugRenderPass &&other) noexcept {
  Swap(other);
}

void DebugRenderPass::operator=(DebugRenderPass &&other) noexcept {
  DebugRenderPass tmp(std::move(other));
  Swap(tmp);
}

void DebugRenderPass::Swap(DebugRenderPass &other) noexcept {
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

void DebugRenderPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer> &) noexcept {
  auto &swapchain = base::Base::Get().GetSwapchain();
  pipeline_.RecordDispatch(
      primary_cmd,
      (swapchain.GetExtent().width * swapchain.GetExtent().height) / 64, 1, 1);
}

TracePrimaryPass::TracePrimaryPass(
    const shader::Loader &trace_primary_shader,
    pipeline_handler::DescriptorPool &pool, BufferBatch<4> geometry_buffers,
    gpu_resources::Buffer *ray_traversal_state,
    gpu_resources::Buffer *per_pixel_state,
    gpu_resources::Buffer *shadow_ray_traversal_state,
    gpu_resources::Buffer *shadow_ray_hash)
    : geometry_buffers_(geometry_buffers),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state),
      shadow_ray_traversal_state_(shadow_ray_traversal_state),
      shadow_ray_hash_(shadow_ray_hash) {
  vk::Extent2D swapchain_ext = base::Base::Get().GetSwapchain().GetExtent();
  uint32_t ray_count = swapchain_ext.width * swapchain_ext.height;
  ray_traversal_state_->RequireProperties(
      {.size = ray_count * kTraversalStateSize});
  per_pixel_state_->RequireProperties({.size = ray_count * kPerPixelStateSize});
  shadow_ray_traversal_state_->RequireProperties(
      {.size = ray_count * kTraversalStateSize});
  shadow_ray_hash_->RequireProperties({.size = ray_count * 4});

  pipeline_handler::DescriptorSet *primary_ray_state =
      trace_primary_shader.GenerateDescriptorSet(pool, 1);
  primary_ray_state->BulkBind(
      std::vector<gpu_resources::Buffer *>{ray_traversal_state_,
                                           per_pixel_state_},
      true);
  pipeline_handler::DescriptorSet *shadow_ray_state =
      trace_primary_shader.GenerateDescriptorSet(pool, 2);
  shadow_ray_state->BulkBind(
      std::vector<gpu_resources::Buffer *>{shadow_ray_traversal_state_,
                                           shadow_ray_hash_},
      true);
  pipeline_ = pipeline_handler::Compute(
      trace_primary_shader,
      {geometry_buffers_.GetDSet(), primary_ray_state, shadow_ray_state});
}

TracePrimaryPass::TracePrimaryPass(TracePrimaryPass &&other) noexcept {
  Swap(other);
}

void TracePrimaryPass::operator=(TracePrimaryPass &&other) noexcept {
  TracePrimaryPass tmp(std::move(other));
  Swap(tmp);
}

void TracePrimaryPass::Swap(TracePrimaryPass &other) noexcept {
  pipeline_.Swap(other.pipeline_);
  std::swap(geometry_buffers_, other.geometry_buffers_);
  std::swap(ray_traversal_state_, other.ray_traversal_state_);
  std::swap(per_pixel_state_, other.per_pixel_state_);
  std::swap(shadow_ray_traversal_state_, other.shadow_ray_traversal_state_);
  std::swap(shadow_ray_hash_, other.shadow_ray_hash_);
}

void TracePrimaryPass::OnPreRecord() {
  gpu_resources::ResourceAccess static_data_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageRead};
  geometry_buffers_.DeclareCommonAccess(static_data_access, GetPassIdx());

  gpu_resources::ResourceAccess ray_trace_state_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite |
          vk::AccessFlagBits2KHR::eShaderStorageRead};
  ray_traversal_state_->DeclareAccess(ray_trace_state_access, GetPassIdx());
  per_pixel_state_->DeclareAccess(ray_trace_state_access, GetPassIdx());
  shadow_ray_traversal_state_->DeclareAccess(ray_trace_state_access,
                                             GetPassIdx());
  shadow_ray_hash_->DeclareAccess(ray_trace_state_access, GetPassIdx());
}

void TracePrimaryPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer> &) noexcept {
  vk::Extent2D extent = base::Base::Get().GetSwapchain().GetExtent();
  uint32_t ray_count = extent.width * extent.height;
  const static uint32_t kRaysPerGroup = 32;
  pipeline_.RecordDispatch(primary_cmd, ray_count / kRaysPerGroup, 1, 1);
}

TraceShadowPass::TraceShadowPass(
    const shader::Loader &trace_shadow_shader,
    pipeline_handler::DescriptorPool &pool, BufferBatch<4> geometry_buffers,
    gpu_resources::Buffer *ray_traversal_state,
    gpu_resources::Buffer *per_pixel_state, gpu_resources::Image *color_target,
    gpu_resources::Image *depth_target,
    gpu_resources::Buffer *shadow_ray_traversal_state,
    gpu_resources::Buffer *shadow_ray_ord)
    : geometry_buffers_(geometry_buffers),
      ray_traversal_state_(ray_traversal_state),
      per_pixel_state_(per_pixel_state), color_target_(color_target),
      depth_target_(depth_target),
      shadow_ray_traversal_state_(shadow_ray_traversal_state),
      shadow_ray_ord_(shadow_ray_ord) {
  pipeline_handler::DescriptorSet *primary_ray_state =
      trace_shadow_shader.GenerateDescriptorSet(pool, 1);
  primary_ray_state->BulkBind(
      std::vector<gpu_resources::Buffer *>{ray_traversal_state_,
                                           per_pixel_state_},
      true);
  pipeline_handler::DescriptorSet *image_target =
      trace_shadow_shader.GenerateDescriptorSet(pool, 2);
  image_target->BulkBind(
      std::vector<gpu_resources::Image *>{color_target_, depth_target_}, true);
  pipeline_handler::DescriptorSet *shadow_ray_state =
      trace_shadow_shader.GenerateDescriptorSet(pool, 3);
  shadow_ray_state->BulkBind(
      std::vector<gpu_resources::Buffer *>{shadow_ray_traversal_state_,
                                           shadow_ray_ord_},
      true);
  pipeline_ = pipeline_handler::Compute(
      trace_shadow_shader, {geometry_buffers_.GetDSet(), primary_ray_state,
                            image_target, shadow_ray_state});
}

TraceShadowPass::TraceShadowPass(TraceShadowPass &&other) noexcept {
  Swap(other);
}

void TraceShadowPass::operator=(TraceShadowPass &&other) noexcept {
  TraceShadowPass tmp(std::move(other));
  Swap(tmp);
}

void TraceShadowPass::Swap(TraceShadowPass &other) noexcept {
  pipeline_.Swap(other.pipeline_);
  std::swap(geometry_buffers_, other.geometry_buffers_);
  std::swap(ray_traversal_state_, other.ray_traversal_state_);
  std::swap(per_pixel_state_, other.per_pixel_state_);
  std::swap(color_target_, other.color_target_);
  std::swap(depth_target_, other.depth_target_);
  std::swap(shadow_ray_traversal_state_, other.shadow_ray_traversal_state_);
  std::swap(shadow_ray_ord_, other.shadow_ray_ord_);
}

void TraceShadowPass::OnPreRecord() {
  gpu_resources::ResourceAccess static_data_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageRead};
  geometry_buffers_.DeclareCommonAccess(static_data_access, GetPassIdx());
  ray_traversal_state_->DeclareAccess(static_data_access, GetPassIdx());
  per_pixel_state_->DeclareAccess(static_data_access, GetPassIdx());
  shadow_ray_ord_->DeclareAccess(static_data_access, GetPassIdx());

  gpu_resources::ResourceAccess shadow_ray_state_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite |
          vk::AccessFlagBits2KHR::eShaderStorageRead};
  shadow_ray_traversal_state_->DeclareAccess(shadow_ray_state_access,
                                             GetPassIdx());
  gpu_resources::ResourceAccess image_target_access{
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite, vk::ImageLayout::eGeneral};
  color_target_->DeclareAccess(image_target_access, GetPassIdx());
  depth_target_->DeclareAccess(image_target_access, GetPassIdx());
}

void TraceShadowPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer> &) noexcept {
  vk::Extent2D extent = base::Base::Get().GetSwapchain().GetExtent();
  uint32_t ray_count = extent.width * extent.height;
  const static uint32_t kRaysPerGroup = 32;
  pipeline_.RecordDispatch(primary_cmd, ray_count / kRaysPerGroup, 1, 1);
}

RayTracer2::RayTracer2(render_data::Mesh const &mesh,
                       render_data::BVH const &bvh) {
  auto device = base::Base::Get().GetContext().GetDevice();
  auto &resource_manager = render_graph_.GetResourceManager();
  auto &swapchain = base::Base::Get().GetSwapchain();

  BufferBatch<4> geometry_buffers;
  PrepareGeometryBuffersTransfer(mesh, bvh, geometry_buffers);
  camera_state_ =
      MainCamera(swapchain.GetExtent().width, swapchain.GetExtent().height);
  shader::Loader raygen_shader(kRayGenShaderPath);
  auto raygen_d_set =
      raygen_shader.GenerateDescriptorSet(render_graph_.GetDescriptorPool(), 0);
  gpu_resources::Buffer *ray_traversal_state = resource_manager.AddBuffer({});
  gpu_resources::Buffer *per_pixel_state = resource_manager.AddBuffer({});
  raygen_ =
      RayGenPass(raygen_shader, raygen_d_set, &camera_state_.GetCameraInfo(),
                 ray_traversal_state, per_pixel_state);
  render_graph_.AddPass(&raygen_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);

  shader::Loader traverse_primary_shader(kTraversePrimaryShaderPath);
  shader::Loader trace_shadow_shader(kTraverseShadowShaderPath);
  geometry_buffers.SetDSet(trace_shadow_shader.GenerateDescriptorSet(
      render_graph_.GetDescriptorPool(), 0));
  gpu_resources::Buffer *shadow_ray_traversal_state =
      resource_manager.AddBuffer({});
  gpu_resources::Buffer *shadow_ray_hash = resource_manager.AddBuffer({});

  trace_primary_ = TracePrimaryPass(
      traverse_primary_shader, render_graph_.GetDescriptorPool(),
      geometry_buffers, ray_traversal_state, per_pixel_state,
      shadow_ray_traversal_state, shadow_ray_hash);
  render_graph_.AddPass(&trace_primary_,
                        vk::PipelineStageFlagBits2KHR::eComputeShader);

  gpu_resources::Buffer *shadow_ray_ord = resource_manager.AddBuffer({});
  shadow_ray_sort_.Apply(
      render_graph_, swapchain.GetExtent().width * swapchain.GetExtent().height,
      shadow_ray_hash, shadow_ray_ord);

  gpu_resources::Image *color_target = resource_manager.AddImage({});
  gpu_resources::Image *depth_target = resource_manager.AddImage({});
  trace_shadow_ = TraceShadowPass(
      trace_shadow_shader, render_graph_.GetDescriptorPool(), geometry_buffers,
      ray_traversal_state, per_pixel_state, color_target, depth_target,
      shadow_ray_traversal_state, shadow_ray_ord);
  render_graph_.AddPass(&trace_shadow_);

  present_ = BlitToSwapchainPass(color_target);
  ready_to_present_ = device.createSemaphore({});
  render_graph_.AddPass(&present_, vk::PipelineStageFlagBits2KHR::eTransfer,
                        ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
}

bool RayTracer2::Draw() {
  camera_state_.Update();
  auto &swapchain = base::Base::Get().GetSwapchain();

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

void RayTracer2::SetCameraPosition(glm::vec3 pos) { camera_state_.SetPos(pos); }

RayTracer2::~RayTracer2() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.waitIdle();
  device.destroySemaphore(ready_to_present_);
}

void RayTracer2::PrepareGeometryBuffersTransfer(render_data::Mesh const &mesh,
                                                render_data::BVH const &bvh,
                                                BufferBatch<4> &buffers) {
  auto &resource_manager = render_graph_.GetResourceManager();
  gpu_resources::Buffer *vertex_pos =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          0, {}, vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer *vertex_ind =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          0, {}, vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer *vertex_nrm =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          mesh.normal.size() * sizeof(mesh.normal[0]),
          {},
          vk::BufferUsageFlagBits::eStorageBuffer});
  gpu_resources::Buffer *bvh_buffer =
      resource_manager.AddBuffer(gpu_resources::BufferProperties{
          bvh.GetNodes().size() * sizeof(bvh.GetNodes()[0]),
          {},
          vk::BufferUsageFlagBits::eStorageBuffer});

  TransferToGPUPass::TransferRequest pos_transfer{
      vertex_pos, (void *)mesh.position.data(),
      mesh.position.size() * sizeof(mesh.position[0])};
  TransferToGPUPass::TransferRequest ind_transfer{
      vertex_ind, (void *)mesh.index.data(),
      mesh.index.size() * sizeof(mesh.index[0])};
  TransferToGPUPass::TransferRequest nrm_transfer{
      vertex_nrm, (void *)mesh.normal.data(),
      mesh.normal.size() * sizeof(mesh.normal[0])};
  TransferToGPUPass::TransferRequest bvh_transfer{
      bvh_buffer, (void *)bvh.GetNodes().data(),
      bvh.GetNodes().size() * sizeof(bvh.GetNodes()[0])};
  transfer_ = TransferToGPUPass(
      {pos_transfer, ind_transfer, nrm_transfer, bvh_transfer},
      resource_manager);

  render_graph_.AddPass(&transfer_, vk::PipelineStageFlagBits2::eTransfer);
  std::array<gpu_resources::Buffer *, 4> buf_arr = {vertex_pos, vertex_ind,
                                                    vertex_nrm, bvh_buffer};
  buffers = BufferBatch<4>(buf_arr);
}
} // namespace examples
