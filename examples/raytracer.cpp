#include "raytracer.h"

#include <vector>

#include "base/base.h"
#include "render_data/bvh.h"
#include "render_data/mesh.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

using utill::Transform;

namespace examples {

const static std::string kColorRTName = "color_target";
const static std::string kDepthRTName = "depth_target";
const static std::string kVertexBufferName = "vertex_buffer";
const static std::string kNormalBufferName = "vertex_normal";
const static std::string kTexcoordBufferName = "vertex_texcoord";
const static std::string kIndexBufferName = "index_buffer";
const static std::string kLightBufferName = "light_buffer";
const static std::string kCameraInfoBufferName = "camera_info";
const static std::string kStagingBufferName = "staging_buffer";
const static std::string kBVHBufferName = "bvh_buffer";
const static float PI = acos(-1);

const static std::vector<std::string> kGeometryBufferNames = {
    kVertexBufferName, kNormalBufferName, kTexcoordBufferName,
    kIndexBufferName};

static render_data::Mesh g_scene_mesh;
static render_data::BVH g_scene_bvh;
static std::vector<glm::vec4> g_light_buffer = {{30, 50, 30, 1.0},
                                                {-30, 50, -30, 1.0}};
static CameraInfo g_camera_info;
static bool g_is_update_camera_transform_ = false;
static int g_move_axis[][2] = {{GLFW_KEY_A, GLFW_KEY_D},
                               {GLFW_KEY_LEFT_SHIFT, GLFW_KEY_SPACE},
                               {GLFW_KEY_S, GLFW_KEY_W}};

using gpu_resources::GetDataSize;

static float GetAxisVal(int axis) {
  if (utill::InputManager::IsKeyPressed(g_move_axis[axis][0])) {
    return -1;
  } else if (utill::InputManager::IsKeyPressed(g_move_axis[axis][1])) {
    return 1;
  }
  return 0;
}

template <typename T>
static size_t FillStagingBuffer(gpu_resources::Buffer* staging_buffer,
                                const std::vector<T>& data,
                                size_t dst_offset) {
  DCHECK(staging_buffer) << "Unexpected null";
  size_t data_byte_size = GetDataSize(data);
  void* map_start = staging_buffer->GetMappingStart();
  DCHECK(map_start) << "Expected buffer memory to be mapped";
  memcpy((char*)map_start + dst_offset, data.data(), data_byte_size);
  dst_offset += data_byte_size;
  return dst_offset;
}

static void RecordCopy(vk::CommandBuffer cmd,
                       gpu_resources::Buffer* staging_buffer,
                       gpu_resources::Buffer* dst_buffer,
                       size_t& staging_offset,
                       size_t size) {
  gpu_resources::Buffer::RecordCopy(cmd, *staging_buffer, *dst_buffer,
                                    staging_offset, 0, size);
  staging_offset += size;
}

void ResourceTransferPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  gpu_resources::Buffer* staging_buffer = GetBuffer(kStagingBufferName);

  if (is_first_record_) {
    is_first_record_ = false;
    size_t staging_offset = 0;
    gpu_resources::Buffer* vertex_gpu_buffer = GetBuffer(kVertexBufferName);
    gpu_resources::Buffer* normal_gpu_buffer = GetBuffer(kNormalBufferName);
    gpu_resources::Buffer* texcoord_gpu_buffer = GetBuffer(kTexcoordBufferName);
    gpu_resources::Buffer* index_gpu_buffer = GetBuffer(kIndexBufferName);
    staging_offset = g_scene_mesh.RecordCopyFromStaging(
        primary_cmd, staging_buffer, vertex_gpu_buffer, normal_gpu_buffer,
        texcoord_gpu_buffer, index_gpu_buffer, staging_offset);

    gpu_resources::Buffer* light_gpu_buffer = GetBuffer(kLightBufferName);
    RecordCopy(primary_cmd, staging_buffer, light_gpu_buffer, staging_offset,
               GetDataSize(g_light_buffer));
    gpu_resources::Buffer* bvh_gpu_buffer = GetBuffer(kBVHBufferName);
    RecordCopy(primary_cmd, staging_buffer, bvh_gpu_buffer, staging_offset,
               GetDataSize(g_scene_bvh.GetNodes()));
  }

  void* camera_buffer_mapping =
      GetBuffer(kCameraInfoBufferName)->GetMappingStart();
  DCHECK(camera_buffer_mapping);
  memcpy(camera_buffer_mapping, &g_camera_info, sizeof(g_camera_info));
}

ResourceTransferPass::ResourceTransferPass() {
  for (const auto& geometry_buffer_name : kGeometryBufferNames) {
    AddBuffer(geometry_buffer_name,
              render_graph::BufferPassBind::TransferDstBuffer());
  }

  AddBuffer(kLightBufferName,
            render_graph::BufferPassBind::TransferDstBuffer());

  AddBuffer(kCameraInfoBufferName,
            render_graph::BufferPassBind::TransferDstBuffer());

  AddBuffer(kBVHBufferName, render_graph::BufferPassBind::TransferDstBuffer());

  AddBuffer(kStagingBufferName,
            render_graph::BufferPassBind::TransferSrcBuffer());
}

void ResourceTransferPass::OnResourcesInitialized() noexcept {
  gpu_resources::Buffer* staging_buffer = GetBuffer(kStagingBufferName);
  size_t fill_offset = 0;
  fill_offset = g_scene_mesh.LoadToStagingBuffer(staging_buffer, fill_offset);
  fill_offset = FillStagingBuffer(staging_buffer, g_light_buffer, fill_offset);
  fill_offset =
      FillStagingBuffer(staging_buffer, g_scene_bvh.GetNodes(), fill_offset);
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
  AddImage(kColorRTName, render_graph::ImagePassBind::ComputeRenderTarget(
                             vk::AccessFlagBits2KHR::eShaderWrite));
  AddImage(kDepthRTName, render_graph::ImagePassBind::ComputeRenderTarget(
                             vk::AccessFlagBits2KHR::eShaderWrite));

  for (const auto& geometry_buffer_name : kGeometryBufferNames) {
    AddBuffer(geometry_buffer_name,
              render_graph::BufferPassBind::ComputeStorageBuffer(
                  vk::AccessFlagBits2KHR::eShaderRead));
  }

  AddBuffer(kLightBufferName,
            render_graph::BufferPassBind::ComputeStorageBuffer(
                vk::AccessFlagBits2KHR::eShaderRead));

  AddBuffer(kCameraInfoBufferName,
            render_graph::BufferPassBind::UniformBuffer(
                vk::PipelineStageFlagBits2KHR::eComputeShader,
                vk::ShaderStageFlagBits::eCompute));

  AddBuffer(kBVHBufferName, render_graph::BufferPassBind::ComputeStorageBuffer(
                                vk::AccessFlagBits2KHR::eShaderRead));

  shader_bindings_.reserve(2 + kGeometryBufferNames.size() + 2);
  shader_bindings_.push_back(&GetImagePassBind(kColorRTName));
  shader_bindings_.push_back(&GetImagePassBind(kDepthRTName));
  for (const auto& geometry_buffer_name : kGeometryBufferNames) {
    shader_bindings_.push_back(&GetBufferPassBind(geometry_buffer_name));
  }
  shader_bindings_.push_back(&GetBufferPassBind(kLightBufferName));
  shader_bindings_.push_back(&GetBufferPassBind(kCameraInfoBufferName));
  shader_bindings_.push_back(&GetBufferPassBind(kBVHBufferName));
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

  g_scene_mesh =
      render_data::Mesh::LoadFromObj("obj/Residential Buildings 001.obj");
  g_scene_bvh =
      render_data::BVH(render_data::BVH::BuildPrimitivesBB(g_scene_mesh));
  g_scene_mesh.ReorderPrimitives(g_scene_bvh.GetPrimitiveOrd());

  resource_manager.AddImage(kColorRTName, {}, {},
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddImage(kDepthRTName, {}, vk::Format::eR32Sfloat,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(kVertexBufferName,
                             GetDataSize(g_scene_mesh.position),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddBuffer(kNormalBufferName,
                             GetDataSize(g_scene_mesh.normal),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddBuffer(kTexcoordBufferName,
                             GetDataSize(g_scene_mesh.tex_coord),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
  resource_manager.AddBuffer(kIndexBufferName, GetDataSize(g_scene_mesh.index),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(kLightBufferName, GetDataSize(g_light_buffer),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(kBVHBufferName,
                             GetDataSize(g_scene_bvh.GetNodes()),
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

  resource_manager.AddBuffer(
      kStagingBufferName,
      GetDataSize(g_scene_mesh.position) + GetDataSize(g_scene_mesh.normal) +
          GetDataSize(g_scene_mesh.tex_coord) +
          GetDataSize(g_scene_mesh.index) + GetDataSize(g_light_buffer) +
          GetDataSize(g_scene_bvh.GetNodes()),
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

void UpdateCameraInfo() {
  auto m_state = utill::InputManager::GetMouseState();
  if (m_state.lmb_state.action == GLFW_PRESS &&
      !g_is_update_camera_transform_) {
    g_is_update_camera_transform_ = true;
    utill::InputManager::SetCursorMode(GLFW_CURSOR_DISABLED);
    m_state = utill::InputManager::GetMouseState();
  } else if (utill::InputManager::IsKeyPressed(GLFW_KEY_ESCAPE) &&
             g_is_update_camera_transform_) {
    utill::InputManager::SetCursorMode(GLFW_CURSOR_NORMAL);
    g_is_update_camera_transform_ = false;
  }
  if (!g_is_update_camera_transform_) {
    return;
  }

  float c_ang_x = m_state.pos_y * (PI / 2);
  float c_ang_y = m_state.pos_x * PI;

  auto& cam_transform = g_camera_info.camera_to_world;
  Transform rotate_y = Transform::RotationY(c_ang_y);
  Transform rotate_x = Transform::Rotation(c_ang_x, rotate_y.GetDirX());
  auto pos = cam_transform.GetPos();
  cam_transform = Transform::Combine(rotate_y, rotate_x);
  glm::vec3 move_dir = cam_transform.GetDirX() * GetAxisVal(0) +
                       cam_transform.GetDirY() * GetAxisVal(1) +
                       cam_transform.GetDirZ() * GetAxisVal(2);
  if (glm::length(move_dir) >= 1) {
    move_dir = glm::normalize(move_dir);
  }
  Transform translate = Transform::Translation(pos + move_dir * 0.1f);
  cam_transform = Transform::Combine(cam_transform, translate);
}

bool RayTracer::Draw() {
  UpdateCameraInfo();
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
