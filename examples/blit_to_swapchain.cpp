#include "blit_to_swapchain.h"

#include "base/base.h"

#include "gpu_resources/physical_image.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace examples {

BlitToSwapchainPass::BlitToSwapchainPass(gpu_resources::Image *render_target)
    : Pass(0), render_target_(render_target) {
  LOG << "Initializing BlitToSwapchainPass";
  DCHECK(render_target != nullptr)
      << "render_target must be valid gpu_resources::Image";
  gpu_resources::ImageProperties render_target_requirements = {};
  render_target_requirements.usage_flags = vk::ImageUsageFlagBits::eTransferSrc;
  render_target_->RequireProperties(render_target_requirements);
}

void BlitToSwapchainPass::OnPreRecord() {
  gpu_resources::ResourceAccess render_target_access{
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferRead,
      vk::ImageLayout::eTransferSrcOptimal};
  render_target_->DeclareAccess(render_target_access, GetPassIdx());
}

void BlitToSwapchainPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer> &) noexcept {
  if (base::Base::Get().GetWindow().GetWindow() == nullptr) {
    return;
  }
  auto &swapchain = base::Base::Get().GetSwapchain();
  gpu_resources::PhysicalImage swapchain_image(
      swapchain.GetImage(swapchain.GetActiveImageInd()), swapchain.GetExtent(),
      swapchain.GetFormat());

  auto pre_blit_barrier = swapchain_image.GenerateBarrier(
      vk::PipelineStageFlagBits2KHR::eTopOfPipe, {},
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eTransferDstOptimal);
  primary_cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, {}, pre_blit_barrier));

  gpu_resources::PhysicalImage::RecordBlit(
      primary_cmd, *render_target_->GetImage(), swapchain_image);

  auto post_blit_barrier = swapchain_image.GenerateBarrier(
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite,
      vk::PipelineStageFlagBits2KHR::eBottomOfPipe, {},
      vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
  primary_cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, {}, post_blit_barrier));

  swapchain_image.Release();
}

} // namespace examples
