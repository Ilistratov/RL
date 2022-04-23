#include "mandelbrot.h"

#include <vulkan/vulkan.hpp>

#include "base/base.h"

#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

namespace examples {
const static std::string kColorTarget = "render_target";

void MandelbrotDrawPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  primary_cmd.pushConstants(compute_pipeline_.GetLayout(),
                            vk::ShaderStageFlagBits::eCompute, 0u,
                            sizeof(PushConstants), &push_constants_);
  compute_pipeline_.RecordDispatch(primary_cmd, swapchain.GetExtent().width / 8,
                                   swapchain.GetExtent().height / 8, 1);
}

MandelbrotDrawPass::MandelbrotDrawPass()
    : Pass(0, vk::PipelineStageFlagBits2KHR::eComputeShader) {
  LOG << "Initializing MandelbrotDrawPass";
  gpu_resources::ResourceUsage rt_usage;
  rt_usage.access = vk::AccessFlagBits2KHR::eShaderWrite;
  rt_usage.stage = vk::PipelineStageFlagBits2KHR::eComputeShader;
  rt_usage.layout = vk::ImageLayout::eGeneral;

  image_binds_[kColorTarget] = render_graph::ImagePassBind(
      rt_usage, vk::ImageUsageFlagBits::eStorage,
      vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
  LOG << "Image binds count " << image_binds_.size();
}

void MandelbrotDrawPass::ReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstants));
  DCHECK(image_binds_.contains(kColorTarget));
  auto rt_bind = image_binds_[kColorTarget];
  compute_pipeline_ = pipeline_handler::Compute({&rt_bind}, pool, {pc_range},
                                                "mandelbrot.spv", "main");
}

void MandelbrotDrawPass::OnResourcesInitialized() noexcept {
  compute_pipeline_.UpdateDescriptorSet({&image_binds_[kColorTarget]});
}

PushConstants& MandelbrotDrawPass::GetPushConstants() {
  return push_constants_;
}

void Mandelbrot::UpdatePushConstants() {
  auto mouse_state = utill::InputManager::GetMouseState();
  auto& swapchain = base::Base::Get().GetSwapchain();
  auto& pc = draw_.GetPushConstants();

  if (mouse_state.lmb_state.action == GLFW_PRESS) {
    float dx = mouse_state.pos_x / swapchain.GetExtent().width;
    float dy = mouse_state.pos_y / swapchain.GetExtent().height;
    float aspect =
        (float)(swapchain.GetExtent().width) / swapchain.GetExtent().height;
    dx = dx - 0.5;
    dy = (dy - 0.5) / aspect;
    dst_x_ = pc.center_x + dx * pc.scale;
    dst_y_ = pc.center_y + dy * pc.scale;
  }
  if (utill::InputManager::IsKeyPressed(GLFW_KEY_Z)) {
    dst_scale_ = pc.scale * 0.5;
  }
  if (utill::InputManager::IsKeyPressed(GLFW_KEY_X)) {
    dst_scale_ = pc.scale * 2;
  }

  pc.s_width = swapchain.GetExtent().width;
  pc.s_height = swapchain.GetExtent().height;
  pc.center_x = pc.center_x * 0.8 + dst_x_ * 0.2;
  pc.center_y = pc.center_y * 0.8 + dst_y_ * 0.2;
  pc.scale = pc.scale * 0.8 + dst_scale_ * 0.2;
}

Mandelbrot::Mandelbrot() : present_(kColorTarget) {
  LOG << "Initializing Renderer";
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  ready_to_present_ = device.createSemaphore({});

  LOG << "Adding resources to RenderGraph";
  render_graph_.GetResourceManager().AddImage(
      kColorTarget, {}, {}, vk::MemoryPropertyFlagBits::eDeviceLocal);
  LOG << "Adding draw pass";
  render_graph_.AddPass(&draw_, {}, {});
  LOG << "Adding present pass";
  render_graph_.AddPass(&present_, ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
  LOG << "Renderer initized";
}

bool Mandelbrot::Draw() {
  UpdatePushConstants();
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

Mandelbrot::~Mandelbrot() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.waitIdle();
  device.destroySemaphore(ready_to_present_);
}

}  // namespace examples
