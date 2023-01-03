#include "raytracer.h"

#include <vcruntime.h>
#include <vector>
#include <vulkan/vulkan_enums.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "pipeline_handler/descriptor_binding.h"
#include "render_data/bvh.h"
#include "render_data/mesh.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

namespace examples {

static render_data::Mesh g_scene_mesh;
static render_data::BVH g_scene_bvh;
static std::vector<glm::vec4> g_light_buffer = {{0, 500, 20, 1.0}};
static MainCamera g_main_camera_state;

using gpu_resources::GetDataSize;

template <typename T>
static void FillStagingBuffer(gpu_resources::Buffer* staging_buffer,
                              const std::vector<T>& data,
                              size_t& dst_offset) {
  DCHECK(staging_buffer) << "Unexpected null";
  size_t data_byte_size = GetDataSize(data);
  void* map_start = staging_buffer->GetBuffer()->GetMappingStart();
  DCHECK(map_start) << "Expected buffer memory to be mapped";
  memcpy((char*)map_start + dst_offset, data.data(), data_byte_size);
  dst_offset += data_byte_size;
}

static void RecordCopyFromStaging(vk::CommandBuffer cmd,
                                  gpu_resources::Buffer* staging_buffer,
                                  gpu_resources::Buffer* dst_buffer,
                                  size_t& staging_offset,
                                  size_t size) {
  gpu_resources::Buffer::RecordCopy(cmd, *staging_buffer, *dst_buffer,
                                    staging_offset, 0, size);
  staging_offset += size;
}

size_t GeometryBuffers::AddBuffersToRenderGraph(
    gpu_resources::ResourceManager& resource_manager) {
  gpu_resources::BufferProperties properties{};
  size_t total_data_size = 0;

  properties.size = GetDataSize(g_scene_mesh.position);
  total_data_size += properties.size;
  position = resource_manager.AddBuffer(properties);

  properties.size = GetDataSize(g_scene_mesh.normal);
  total_data_size += properties.size;
  normal = resource_manager.AddBuffer(properties);

  properties.size = GetDataSize(g_scene_mesh.tex_coord);
  total_data_size += properties.size;
  tex_coord = resource_manager.AddBuffer(properties);

  properties.size = GetDataSize(g_scene_mesh.index);
  total_data_size += properties.size;
  index = resource_manager.AddBuffer(properties);

  properties.size = GetDataSize(g_light_buffer);
  total_data_size += properties.size;
  light = resource_manager.AddBuffer(properties);

  properties.size = GetDataSize(g_scene_bvh.GetNodes());
  total_data_size += properties.size;
  bvh = resource_manager.AddBuffer(properties);
  return total_data_size;
}

void GeometryBuffers::AddCommonRequierment(
    gpu_resources::BufferProperties requierment) {
  position->RequireProperties(requierment);
  normal->RequireProperties(requierment);
  tex_coord->RequireProperties(requierment);
  index->RequireProperties(requierment);
  light->RequireProperties(requierment);
  bvh->RequireProperties(requierment);
}

void GeometryBuffers::DeclareCommonAccess(gpu_resources::ResourceAccess access,
                                          uint32_t pass_idx) {
  position->DeclareAccess(access, pass_idx);
  normal->DeclareAccess(access, pass_idx);
  tex_coord->DeclareAccess(access, pass_idx);
  index->DeclareAccess(access, pass_idx);
  light->DeclareAccess(access, pass_idx);
  bvh->DeclareAccess(access, pass_idx);
}

GeometryBindings::GeometryBindings(GeometryBuffers buffers,
                                   vk::ShaderStageFlags access_stage) {
  vk::DescriptorType type = vk::DescriptorType::eStorageBuffer;
  position = pipeline_handler::BufferDescriptorBinding(buffers.position, type,
                                                       access_stage);
  normal = pipeline_handler::BufferDescriptorBinding(buffers.normal, type,
                                                     access_stage);
  tex_coord = pipeline_handler::BufferDescriptorBinding(buffers.tex_coord, type,
                                                        access_stage);
  index = pipeline_handler::BufferDescriptorBinding(buffers.index, type,
                                                    access_stage);
  light = pipeline_handler::BufferDescriptorBinding(buffers.light, type,
                                                    access_stage);
  bvh = pipeline_handler::BufferDescriptorBinding(buffers.bvh, type,
                                                  access_stage);
}

ResourceTransferPass::ResourceTransferPass(
    GeometryBuffers geometry,
    gpu_resources::Buffer* staging_buffer,
    gpu_resources::Buffer* camera_info)
    : geometry_(geometry),
      staging_buffer_(staging_buffer),
      camera_info_(camera_info) {
  gpu_resources::BufferProperties required_transfer_src_properties{};
  required_transfer_src_properties.memory_flags =
      vk::MemoryPropertyFlagBits::eHostVisible;
  required_transfer_src_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc;
  staging_buffer_->RequireProperties(required_transfer_src_properties);

  gpu_resources::BufferProperties required_transfer_dst_properties{};
  required_transfer_dst_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferDst;
  geometry.AddCommonRequierment(required_transfer_dst_properties);

  gpu_resources::BufferProperties required_camera_info_properties{};
  required_camera_info_properties.memory_flags =
      vk::MemoryPropertyFlagBits::eHostVisible;
  required_camera_info_properties.size = sizeof(CameraInfo);
  camera_info_->RequireProperties(required_camera_info_properties);
}

void ResourceTransferPass::OnResourcesInitialized() noexcept {
  size_t fill_offset = 0;
  FillStagingBuffer(staging_buffer_, g_scene_mesh.position, fill_offset);
  FillStagingBuffer(staging_buffer_, g_scene_mesh.normal, fill_offset);
  FillStagingBuffer(staging_buffer_, g_scene_mesh.tex_coord, fill_offset);
  FillStagingBuffer(staging_buffer_, g_scene_mesh.index, fill_offset);

  FillStagingBuffer(staging_buffer_, g_light_buffer, fill_offset);
  FillStagingBuffer(staging_buffer_, g_scene_bvh.GetNodes(), fill_offset);

  auto device = base::Base::Get().GetContext().GetDevice();
  device.flushMappedMemoryRanges(
      staging_buffer_->GetBuffer()->GetMappedMemoryRange());
}

void ResourceTransferPass::OnPreRecord() {
  vk::PipelineStageFlags2KHR pass_stage =
      vk::PipelineStageFlagBits2KHR::eTransfer;
  gpu_resources::ResourceAccess transfer_src_access{};
  transfer_src_access.access_flags = vk::AccessFlagBits2KHR::eTransferRead;
  transfer_src_access.stage_flags = pass_stage;
  staging_buffer_->DeclareAccess(transfer_src_access, GetPassIdx());

  gpu_resources::ResourceAccess transfer_dst_access{};
  transfer_dst_access.access_flags = vk::AccessFlagBits2KHR::eTransferWrite;
  transfer_dst_access.stage_flags = pass_stage;
  geometry_.DeclareCommonAccess(transfer_dst_access, GetPassIdx());
}

void ResourceTransferPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  if (is_first_record_) {
    is_first_record_ = false;
    size_t staging_offset = 0;
    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.position,
                          staging_offset, GetDataSize(g_scene_mesh.position));
    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.normal,
                          staging_offset, GetDataSize(g_scene_mesh.normal));
    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.tex_coord,
                          staging_offset, GetDataSize(g_scene_mesh.tex_coord));
    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.index,
                          staging_offset, GetDataSize(g_scene_mesh.index));

    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.light,
                          staging_offset, GetDataSize(g_light_buffer));
    RecordCopyFromStaging(primary_cmd, staging_buffer_, geometry_.bvh,
                          staging_offset, GetDataSize(g_scene_bvh.GetNodes()));
  }

  void* camera_buffer_mapping = camera_info_->GetBuffer()->GetMappingStart();
  DCHECK(camera_buffer_mapping);
  memcpy(camera_buffer_mapping, &g_main_camera_state.GetCameraInfo(),
         sizeof(CameraInfo));
}

RaytracerPass::RaytracerPass(GeometryBuffers geometry,
                             gpu_resources::Image* color_target,
                             gpu_resources::Image* depth_target,
                             gpu_resources::Buffer* camera_info)
    : geometry_(geometry),
      color_target_(color_target),
      depth_target_(depth_target),
      camera_info_(camera_info) {
  gpu_resources::BufferProperties requeired_buffer_propertires{};
  requeired_buffer_propertires.memory_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;
  requeired_buffer_propertires.usage_flags =
      vk::BufferUsageFlagBits::eStorageBuffer;
  geometry_.AddCommonRequierment(requeired_buffer_propertires);

  gpu_resources::ImageProperties required_image_prperties{};
  required_image_prperties.memory_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;
  required_image_prperties.usage_flags = vk::ImageUsageFlagBits::eStorage;
  color_target_->RequireProperties(required_image_prperties);
  depth_target_->RequireProperties(required_image_prperties);

  gpu_resources::BufferProperties requeired_camera_info_propertires{};
  requeired_camera_info_propertires.size = sizeof(CameraInfo);
  requeired_camera_info_propertires.usage_flags =
      vk::BufferUsageFlagBits::eUniformBuffer;
  camera_info_->RequireProperties(requeired_camera_info_propertires);

  vk::ShaderStageFlags pass_shader_stage = vk::ShaderStageFlagBits::eCompute;
  geometry_bindings_ = GeometryBindings(geometry_, pass_shader_stage);
  color_target_binding_ = pipeline_handler::ImageDescriptorBinding(
      color_target_, vk::DescriptorType::eStorageImage, pass_shader_stage,
      vk::ImageLayout::eGeneral);
  depth_target_binding_ = pipeline_handler::ImageDescriptorBinding(
      depth_target_, vk::DescriptorType::eStorageImage, pass_shader_stage,
      vk::ImageLayout::eGeneral);

  camera_info_binding_ = pipeline_handler::BufferDescriptorBinding(
      camera_info_, vk::DescriptorType::eUniformBuffer, pass_shader_stage);
}

void RaytracerPass::OnReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  pipeline_ = pipeline_handler::Compute(
      {
          &color_target_binding_,
          &depth_target_binding_,
          &geometry_bindings_.position,
          &geometry_bindings_.normal,
          &geometry_bindings_.tex_coord,
          &geometry_bindings_.index,
          &geometry_bindings_.light,
          &camera_info_binding_,
          &geometry_bindings_.bvh,
      },
      pool, {}, "raytrace.spv", "main");
}

void RaytracerPass::OnPreRecord() {
  gpu_resources::ResourceAccess scene_resource_access{};
  scene_resource_access.access_flags = vk::AccessFlagBits2KHR::eShaderRead;
  scene_resource_access.stage_flags =
      vk::PipelineStageFlagBits2KHR::eComputeShader;
  geometry_.DeclareCommonAccess(scene_resource_access, GetPassIdx());
  scene_resource_access.layout = vk::ImageLayout::eGeneral;
  color_target_->DeclareAccess(scene_resource_access, GetPassIdx());
  depth_target_->DeclareAccess(scene_resource_access, GetPassIdx());
}

void RaytracerPass::OnRecord(vk::CommandBuffer primary_cmd,
                             const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  pipeline_.RecordDispatch(primary_cmd, swapchain.GetExtent().width / 8,
                           swapchain.GetExtent().height / 8, 1);
}

RayTracer::RayTracer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  ready_to_present_ = device.createSemaphore({});
  auto& resource_manager = render_graph_.GetResourceManager();

  g_scene_mesh =
      render_data::Mesh::LoadFromObj("../assets/objects/serpentine_city.obj");
  g_scene_bvh =
      render_data::BVH(render_data::BVH::BuildPrimitivesBB(g_scene_mesh));
  g_scene_mesh.ReorderPrimitives(g_scene_bvh.GetPrimitiveOrd());

  gpu_resources::BufferProperties buffer_properties{};
  GeometryBuffers geometry{};
  buffer_properties.size = geometry.AddBuffersToRenderGraph(resource_manager);
  gpu_resources::Buffer* staging_buffer =
      resource_manager.AddBuffer(buffer_properties);

  gpu_resources::ImageProperties image_properties{};
  gpu_resources::Image* color_target =
      resource_manager.AddImage(image_properties);
  image_properties.format = vk::Format::eR32Sfloat;
  gpu_resources::Image* depth_target =
      resource_manager.AddImage(image_properties);

  buffer_properties.size = sizeof(CameraInfo);
  gpu_resources::Buffer* camera_info =
      resource_manager.AddBuffer(buffer_properties);

  resource_transfer_ =
      ResourceTransferPass(geometry, staging_buffer, camera_info);
  render_graph_.AddPass(&resource_transfer_);

  raytrace_ = RaytracerPass(geometry, color_target, depth_target, camera_info);
  render_graph_.AddPass(&raytrace_);

  present_ = BlitToSwapchainPass(color_target);
  render_graph_.AddPass(&present_, vk::PipelineStageFlagBits2KHR::eTransfer,
                        ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
  g_main_camera_state =
      MainCamera(swapchain.GetExtent().width, swapchain.GetExtent().height);
}

bool RayTracer::Draw() {
  g_main_camera_state.Update();
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

RayTracer::~RayTracer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.waitIdle();
  device.destroySemaphore(ready_to_present_);
}

}  // namespace examples
