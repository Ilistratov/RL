#include "mandelbrot.h"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "base/base.h"

#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "pipeline_handler/descriptor_binding.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

namespace examples {

MandelbrotDrawPass::MandelbrotDrawPass(gpu_resources::Image* render_target)
    : Pass(0), render_target_(render_target) {
  LOG << "Initializing MandelbrotDrawPass";
  gpu_resources::ImageProperties render_target_requierments{};
  render_target_requierments.usage_flags = vk::ImageUsageFlagBits::eStorage;
  render_target_->RequireProperties(render_target_requierments);

  render_target_binding_ = pipeline_handler::ImageDescriptorBinding(
      render_target_, vk::DescriptorType::eStorageImage,
      vk::ShaderStageFlagBits::eCompute, vk::ImageLayout::eGeneral);
}

void MandelbrotDrawPass::OnReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstants));
  compute_pipeline_ = pipeline_handler::Compute(
      {&render_target_binding_}, pool, {pc_range}, "mandelbrot.spv", "main");
}

void MandelbrotDrawPass::OnPreRecord() {
  gpu_resources::ResourceAccess render_target_access;
  render_target_access.access_flags = vk::AccessFlagBits2KHR::eShaderWrite;
  render_target_access.stage_flags =
      vk::PipelineStageFlagBits2KHR::eComputeShader;
  render_target_access.layout = vk::ImageLayout::eGeneral;
  render_target_->DeclareAccess(render_target_access, GetPassIdx());
}

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

Mandelbrot::Mandelbrot() {
  LOG << "Initializing Renderer";
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  ready_to_present_ = device.createSemaphore({});

  LOG << "Adding resources to RenderGraph";
  gpu_resources::ImageProperties render_target_propertires;
  render_target_propertires.memory_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;
  render_target_ =
      render_graph_.GetResourceManager().AddImage(render_target_propertires);

  draw_ = MandelbrotDrawPass(render_target_);
  LOG << "Adding draw pass";
  render_graph_.AddPass(&draw_, vk::PipelineStageFlagBits2KHR::eComputeShader,
                        {});
  LOG << "Adding present pass";
  present_ = BlitToSwapchainPass(render_target_);
  render_graph_.AddPass(&present_, vk::PipelineStageFlagBits2KHR::eTransfer,
                        ready_to_present_,
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
