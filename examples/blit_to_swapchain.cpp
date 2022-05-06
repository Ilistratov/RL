#include "blit_to_swapchain.h"

#include "base/base.h"

#include "utill/logger.h"

namespace examples {

BlitToSwapchainPass::BlitToSwapchainPass(const std::string& render_target_name)
    : Pass(0, vk::PipelineStageFlagBits2KHR::eTransfer),
      render_target_name_(render_target_name) {
  LOG << "Initializing BlitToSwapchainPass";
  image_binds_[render_target_name_] = render_graph::ImagePassBind(
      gpu_resources::ResourceUsage{vk::PipelineStageFlagBits2KHR::eTransfer,
                                   vk::AccessFlagBits2KHR::eTransferRead,
                                   vk::ImageLayout::eTransferSrcOptimal},
      vk::ImageUsageFlagBits::eTransferSrc);
  LOG << "Image binds count " << image_binds_.size();
}

void BlitToSwapchainPass::OnRecord(
    vk::CommandBuffer primary_cmd,
    const std::vector<vk::CommandBuffer>&) noexcept {
  auto& swapchain = base::Base::Get().GetSwapchain();
  gpu_resources::Image swapchain_image(
      swapchain.GetImage(swapchain.GetActiveImageInd()), swapchain.GetExtent(),
      swapchain.GetFormat());

  auto pre_blit_barrier = swapchain_image.GetBarrier(
      vk::PipelineStageFlagBits2KHR::eTopOfPipe, {},
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eTransferDstOptimal);
  primary_cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, {}, pre_blit_barrier));

  gpu_resources::Image::RecordBlit(
      primary_cmd, *image_binds_[render_target_name_].GetBoundImage(),
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

}  // namespace examples
