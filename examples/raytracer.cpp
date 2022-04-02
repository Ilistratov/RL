#include "raytracer.h"

#include "base/base.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

namespace examples {

const static std::string kColorRTName = "color_target";
const static std::string kDepthRTName = "depth_target";
const static std::string kVertexBufferName = "vertex_buffer";
const static std::string kIndexBufferName = "index_buffer";
const static std::string kLightBufferName = "light_buffer";
const static std::string kCameraInfoBufferName = "camera_info";
const static std::string kStagingBufferName = "staging_buffer";

std::vector<glm::vec4> g_vertex_buffer = {
    {0.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {1.0, 1.0, 0.0, 1.0},
    {0.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0},
};
std::vector<uint32_t> g_index_buffer = {0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7};
std::vector<glm::vec4> g_light_buffer = {{0.5, 2, 0.5, 1.0}};
static CameraInfo g_camera_info;

template <typename T>
static inline size_t GetDataSize(const std::vector<T>& data) {
  return sizeof(T) * data.size();
}

template <typename T>
static void FillStagingBuffer(gpu_resources::LogicalBuffer* staging_buffer,
                              const std::vector<T>& data,
                              size_t& dst_offset) {
  DCHECK(staging_buffer) << "Unexpected null";
  size_t data_byte_size = GetDataSize(data);
  void* map_start = staging_buffer->GetMappingStart();
  DCHECK(map_start) << "Expected buffer memory to be mapped";
  memcpy((char*)map_start + dst_offset, data.data(), data_byte_size);
  dst_offset += data_byte_size;
}

static void RecordCopy(vk::CommandBuffer cmd,
                       gpu_resources::LogicalBuffer* staging_buffer,
                       gpu_resources::LogicalBuffer* dst_buffer,
                       size_t& staging_offset,
                       size_t size) {
  gpu_resources::PhysicalBuffer::RecordCopy(
      cmd, staging_buffer->GetPhysicalBuffer(), dst_buffer->GetPhysicalBuffer(),
      staging_offset, 0, size);
  staging_offset += size;
}

void ResourceTransferPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  gpu_resources::LogicalBuffer* staging_buffer =
      buffer_binds_[kStagingBufferName].GetBoundBuffer();
  size_t staging_offset = 0;

  if (is_first_record_) {
    is_first_record_ = false;
    gpu_resources::LogicalBuffer* vertex_logical_buffer =
        buffer_binds_[kVertexBufferName].GetBoundBuffer();
    RecordCopy(primary_cmd, staging_buffer, vertex_logical_buffer,
               staging_offset, GetDataSize(g_vertex_buffer));

    gpu_resources::LogicalBuffer* index_logical_buffer =
        buffer_binds_[kIndexBufferName].GetBoundBuffer();
    RecordCopy(primary_cmd, staging_buffer, index_logical_buffer,
               staging_offset, GetDataSize(g_index_buffer));

    gpu_resources::LogicalBuffer* light_logical_buffer =
        buffer_binds_[kLightBufferName].GetBoundBuffer();
    RecordCopy(primary_cmd, staging_buffer, light_logical_buffer,
               staging_offset, GetDataSize(g_index_buffer));
  }

  void* camera_buffer_mapping =
      buffer_binds_[kCameraInfoBufferName].GetBoundBuffer()->GetMappingStart();
  DCHECK(camera_buffer_mapping);
  memcpy(camera_buffer_mapping, &g_camera_info, sizeof(g_camera_info));
}

ResourceTransferPass::ResourceTransferPass() {
  buffer_binds_[kVertexBufferName] =
      render_graph::BufferPassBind::TransferDstBuffer();
  buffer_binds_[kIndexBufferName] =
      render_graph::BufferPassBind::TransferDstBuffer();
  buffer_binds_[kLightBufferName] =
      render_graph::BufferPassBind::TransferDstBuffer();

  buffer_binds_[kCameraInfoBufferName] =
      render_graph::BufferPassBind::TransferDstBuffer();

  buffer_binds_[kStagingBufferName] =
      render_graph::BufferPassBind::TransferSrcBuffer();
}

void ResourceTransferPass::OnResourcesInitialized() noexcept {
  gpu_resources::LogicalBuffer* staging_buffer =
      buffer_binds_[kStagingBufferName].GetBoundBuffer();
  size_t fill_offset = 0;
  FillStagingBuffer(staging_buffer, g_vertex_buffer, fill_offset);
  FillStagingBuffer(staging_buffer, g_index_buffer, fill_offset);
  FillStagingBuffer(staging_buffer, g_light_buffer, fill_offset);
  auto device = base::Base::Get().GetContext().GetDevice();
  device.flushMappedMemoryRanges(staging_buffer->GetMappedMemoryRange());
}

void RaytracerPass::OnRecord(vk::CommandBuffer primary_cmd,
                             const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  pipeline_.RecordDispatch(primary_cmd, swapchain.GetExtent().width / 8,
                           swapchain.GetExtent().height / 8, 1);
}

RaytracerPass::RaytracerPass() {
  image_binds_[kColorRTName] = render_graph::ImagePassBind::ComputeRenderTarget(
      vk::AccessFlagBits2KHR::eShaderWrite);
  image_binds_[kDepthRTName] = render_graph::ImagePassBind::ComputeRenderTarget(
      vk::AccessFlagBits2KHR::eShaderWrite);

  buffer_binds_[kVertexBufferName] =
      render_graph::BufferPassBind::ComputeStorageBuffer(
          vk::AccessFlagBits2KHR::eShaderRead);
  buffer_binds_[kIndexBufferName] =
      render_graph::BufferPassBind::ComputeStorageBuffer(
          vk::AccessFlagBits2KHR::eShaderRead);
  buffer_binds_[kLightBufferName] =
      render_graph::BufferPassBind::ComputeStorageBuffer(
          vk::AccessFlagBits2KHR::eShaderRead);

  buffer_binds_[kCameraInfoBufferName] =
      render_graph::BufferPassBind::UniformBuffer(
          vk::PipelineStageFlagBits2KHR::eComputeShader,
          vk::ShaderStageFlagBits::eCompute);
  shader_bindings_ = {
      &image_binds_[kColorRTName],       &image_binds_[kDepthRTName],
      &buffer_binds_[kVertexBufferName], &buffer_binds_[kIndexBufferName],
      &buffer_binds_[kLightBufferName],  &buffer_binds_[kCameraInfoBufferName],
  };
}

void RaytracerPass::ReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  pipeline_ = pipeline_handler::Compute(shader_bindings_, pool, {},
                                        "raytrace.spv", "main");
}

void RaytracerPass::OnResourcesInitialized() noexcept {
  pipeline_.UpdateDescriptorSet(shader_bindings_);
}

RayTracer::RayTracer() : present_(kColorRTName) {
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  ready_to_present_ = device.createSemaphore({});
  auto& resource_manager = render_graph_.GetResourceManager();

  resource_manager.AddImage(kColorRTName, {}, {},
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddImage(kDepthRTName, {}, vk::Format::eR32Sfloat,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(kVertexBufferName, GetDataSize(g_vertex_buffer),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddBuffer(kIndexBufferName, GetDataSize(g_index_buffer),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddBuffer(kLightBufferName, GetDataSize(g_vertex_buffer),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(kStagingBufferName,
                             GetDataSize(g_vertex_buffer) +
                                 GetDataSize(g_index_buffer) +
                                 GetDataSize(g_vertex_buffer),
                             vk::MemoryPropertyFlagBits::eHostVisible);
  resource_manager.AddBuffer(kCameraInfoBufferName, sizeof(CameraInfo),
                             vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent);

  render_graph_.AddPass(&resource_transfer_);
  render_graph_.AddPass(&raytrace_);
  render_graph_.AddPass(&present_, ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
}

void UpdateCameraInfo() {}

bool RayTracer::Draw() {
  auto& swapchain = base::Base::Get().GetSwapchain();

  g_camera_info.screen_width = swapchain.GetExtent().width;
  g_camera_info.screen_height = swapchain.GetExtent().height;
  g_camera_info.aspect =
      float(g_camera_info.screen_width) / g_camera_info.screen_height;

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
