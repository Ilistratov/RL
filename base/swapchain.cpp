#include "base/swapchain.h"

#include "base/base.h"
#include "utill/logger.h"

namespace base {

vk::Format Swapchain::PickFormat(vk::SurfaceKHR surface) const {
  auto device = base::Base::Get().GetContext().GetPhysicalDevice();
  auto formats = device.getSurfaceFormatsKHR(surface);
  if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
    return formats[0].format;
  }
  for (const auto& format : formats) {
    if (format.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
      continue;
    }
    if (format.format == vk::Format::eB8G8R8A8Unorm) {
      return format.format;
    }
  }
  LOG(ERROR) << "Failed to pick swapchain format";
  assert(false);
  return vk::Format::eUndefined;
}

vk::Extent2D Swapchain::PickExtent(
    vk::SurfaceCapabilitiesKHR surface_capabilities) const {
  assert(surface_capabilities.currentExtent.width != UINT32_MAX);
  assert(surface_capabilities.currentExtent.height != UINT32_MAX);
  return surface_capabilities.currentExtent;
}

vk::SwapchainCreateInfoKHR Swapchain::GetCreateInfo() const {
  auto surface = Base::Get().GetWindow().GetSurface();
  auto physical_device = Base::Get().GetContext().GetPhysicalDevice();
  auto surface_capabilities =
      physical_device.getSurfaceCapabilitiesKHR(surface);
  vk::SwapchainCreateInfoKHR swapchain_info(
      vk::SwapchainCreateFlagsKHR{}, surface, 2, PickFormat(surface),
      vk::ColorSpaceKHR::eSrgbNonlinear, PickExtent(surface_capabilities), 1,
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferDst,
      vk::SharingMode::eExclusive, {}, surface_capabilities.currentTransform);
  swapchain_info.presentMode = vk::PresentModeKHR::eFifo;
  return swapchain_info;
}

void Swapchain::Create() {
  assert(!swapchain_);
  auto device = Base::Get().GetContext().GetDevice();
  auto swapchain_info = GetCreateInfo();
  swapchain_ = device.createSwapchainKHR(swapchain_info);
  format_ = swapchain_info.imageFormat;
  extent_ = swapchain_info.imageExtent;
  images_ = device.getSwapchainImagesKHR(swapchain_);
  image_avaliable_ = device.createSemaphore(vk::SemaphoreCreateInfo{});
}

void Swapchain::Destroy() {
  auto device = Base::Get().GetContext().GetDevice();
  images_.clear();
  device.destroySemaphore(image_avaliable_);
  device.destroySwapchainKHR(swapchain_);
}

vk::Extent2D Swapchain::GetExtent() const noexcept {
  return extent_;
}

vk::Format Swapchain::GetFormat() const noexcept {
  return format_;
}

vk::Image Swapchain::GetImage(uint32_t image_ind) const noexcept {
  return images_[image_ind];
}

const static uint64_t SWAPCHAIN_PRESENT_TIMEOUT_NSEC = 5'000'000'000;

bool Swapchain::AcquireNextImage() {
  if (active_image_ind_ != UINT32_MAX) {
    LOG(WARNING) << "acquireNextImage called before previously acquired image "
                    "was presented, keeping old activeImageInd";
    return true;
  }

  auto device = base::Base::Get().GetContext().GetDevice();
  auto res =
      device.acquireNextImageKHR(swapchain_, SWAPCHAIN_PRESENT_TIMEOUT_NSEC,
                                 image_avaliable_, {}, &active_image_ind_);
  if (res == vk::Result::eSuccess) {
    return true;
  } else if (res == vk::Result::eSuboptimalKHR) {
    active_image_ind_ = UINT32_MAX;
    return false;
  }

  LOG(ERROR) << "Failed to acquire swapchain image: " << vk::to_string(res);
  assert(false);
  return false;
}

uint32_t Swapchain::GetActiveImageInd() const noexcept {
  return active_image_ind_;
}

vk::Semaphore Swapchain::GetImageAvaliableSemaphore() const noexcept {
  return image_avaliable_;
}

}  // namespace base
