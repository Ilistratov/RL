#include "mandelbrot.h"

#include <vulkan/vulkan.hpp>

#include "base/base.h"

#include "utill/logger.h"

namespace examples {
const std::string kRT_NAME = "render_target";

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
  LOG(INFO) << "Initializing MandelbrotDrawPass";
  gpu_resources::ResourceUsage rt_usage;
  rt_usage.access = vk::AccessFlagBits2KHR::eShaderWrite;
  rt_usage.stage = vk::PipelineStageFlagBits2KHR::eComputeShader;
  rt_usage.layout = vk::ImageLayout::eGeneral;

  image_binds_[kRT_NAME] = render_graph::ImagePassBind(
      rt_usage, vk::ImageUsageFlagBits::eStorage,
      vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
  LOG(INFO) << "Image binds count " << image_binds_.size();
}

void MandelbrotDrawPass::ReserveDescriptorSets(
    pipeline_handler::DescriptorPool& pool) noexcept {
  vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstants));
  assert(image_binds_.contains(kRT_NAME));
  auto rt_bind = image_binds_[kRT_NAME];
  compute_pipeline_ = pipeline_handler::Compute({&rt_bind}, pool, {pc_range},
                                                "mandelbrot.spv", "main");
}

void MandelbrotDrawPass::OnResourcesInitialized() noexcept {
  compute_pipeline_.UpdateDescriptorSet({&image_binds_[kRT_NAME]});
}

PushConstants& MandelbrotDrawPass::GetPushConstants() {
  return push_constants_;
}

SwapchainPresentPass::SwapchainPresentPass()
    : Pass(0, vk::PipelineStageFlagBits2KHR::eTransfer) {
  LOG(INFO) << "Initializing SwapchainPresentPass";
  image_binds_[kRT_NAME] = render_graph::ImagePassBind(
      gpu_resources::ResourceUsage{vk::PipelineStageFlagBits2KHR::eTransfer,
                                   vk::AccessFlagBits2KHR::eTransferRead,
                                   vk::ImageLayout::eTransferSrcOptimal},
      vk::ImageUsageFlagBits::eTransferSrc);
  LOG(INFO) << "Image binds count " << image_binds_.size();
}

void SwapchainPresentPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  gpu_resources::PhysicalImage swapchain_image(
      swapchain.GetImage(swapchain.GetActiveImageInd()), swapchain.GetExtent(),
      swapchain.GetFormat());

  auto pre_blit_barrier = swapchain_image.GetBarrier(
      vk::PipelineStageFlagBits2KHR::eTopOfPipe, {},
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eTransferDstOptimal);
  primary_cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, {}, pre_blit_barrier));

  gpu_resources::PhysicalImage::RecordBlit(
      primary_cmd, image_binds_[kRT_NAME].GetBoundImage()->GetPhysicalImage(),
      swapchain_image);

  auto post_blit_barrier = swapchain_image.GetBarrier(
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite,
      vk::PipelineStageFlagBits2KHR::eBottomOfPipe, {},
      vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
  primary_cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, {}, post_blit_barrier));

  swapchain_image.Release();
}

void Mandelbrot::UpdatePushConstants() {
  auto& swapchain = base::Base::Get().GetSwapchain();
  auto& window = base::Base::Get().GetWindow();
  PushConstants& pc = draw_.GetPushConstants();
  pc.s_width = swapchain.GetExtent().width;
  pc.s_height = swapchain.GetExtent().height;
  double xpos, ypos;
  glfwGetCursorPos(window.GetWindow(), &xpos, &ypos);
  double rat = -1.0 / swapchain.GetExtent().height;
  double aspect =
      double(swapchain.GetExtent().width) / swapchain.GetExtent().height;
  pc.center_x = (xpos * rat + aspect) * 2;
  pc.center_y = (ypos * rat + 0.5) * 2;
}

Mandelbrot::Mandelbrot() {
  LOG(INFO) << "Initializing Renderer";
  auto device = base::Base::Get().GetContext().GetDevice();
  auto& swapchain = base::Base::Get().GetSwapchain();
  ready_to_present_ = device.createSemaphore({});

  LOG(INFO) << "Adding resources to RenderGraph";
  render_graph_.GetResourceManager().AddImage(
      kRT_NAME, {}, {}, vk::MemoryPropertyFlagBits::eDeviceLocal);
  LOG(INFO) << "Adding draw pass";
  render_graph_.AddPass(&draw_, {}, {});
  LOG(INFO) << "Adding present pass";
  render_graph_.AddPass(&present_, ready_to_present_,
                        swapchain.GetImageAvaliableSemaphore());
  render_graph_.Init();
  LOG(INFO) << "Renderer initized";
}

bool Mandelbrot::Draw() {
  UpdatePushConstants();
  auto& swapchain = base::Base::Get().GetSwapchain();
  if (!swapchain.AcquireNextImage()) {
    LOG(ERROR) << "Failed to acquire next image";
    return false;
  }
  swapchain.GetActiveImageInd();
  render_graph_.RenderFrame();
  if (swapchain.Present(ready_to_present_) != vk::Result::eSuccess) {
    LOG(ERROR) << "Failed to present";
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
