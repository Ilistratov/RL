#pragma once

#include <vulkan/vulkan.hpp>

namespace base {

class Swapchain {
  vk::SwapchainKHR swapchain_;
  vk::Format format_ = vk::Format::eUndefined;
  vk::Extent2D extent_ = vk::Extent2D{0, 0};
  std::vector<vk::Image> images_;
  vk::Semaphore image_avaliable_;
  uint32_t active_image_ind_ = UINT32_MAX;

  vk::Format PickFormat(vk::SurfaceKHR surface) const;
  vk::Extent2D PickExtent(
      vk::SurfaceCapabilitiesKHR surface_capabilities) const;
  vk::SwapchainCreateInfoKHR GetCreateInfo() const;

 public:
  Swapchain() = default;
  Swapchain(const Swapchain&) = delete;
  void operator=(const Swapchain&) = delete;

  void Create();
  void Destroy();

  vk::Extent2D GetExtent() const noexcept;
  vk::Format GetFormat() const noexcept;
  vk::Image GetImage(uint32_t image_ind) const noexcept;
  uint32_t GetImageCount() const;

  bool AcquireNextImage();
  uint32_t GetActiveImageInd() const noexcept;
  vk::Semaphore GetImageAvaliableSemaphore() const noexcept;
  vk::Result Present(vk::Semaphore semaphore_to_wait);
};

}  // namespace base
