#pragma once

#include <vulkan/vulkan.hpp>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/physical_image.h"

namespace gpu_resources {

class LogicalImage {
  PhysicalImage image_;
  AccessSyncManager access_manager_;
  std::vector<AccessDependency> dependencies_;

  vk::Extent2D extent_;
  vk::Format format_;
  vk::MemoryPropertyFlags memory_flags_;
  vk::ImageUsageFlags usage_flags_;
  MemoryBlock* memory_ = nullptr;

  LogicalImage(vk::Image image,
               vk::Extent2D extent,
               vk::Format format,
               vk::MemoryPropertyFlags memory_properties);

 public:
  LogicalImage() = default;

  static LogicalImage CreateStorageImage(vk::Extent2D = {0, 0});
  static LogicalImage CreateSwapchainImage(uint32_t swapchain_image_ind);

  LogicalImage(const LogicalImage&) = delete;
  void operator=(const LogicalImage&) = delete;

  LogicalImage(LogicalImage&& other) noexcept;
  void operator=(LogicalImage&& other) noexcept;
  void Swap(LogicalImage& other) noexcept;

  void CreateImage();
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindBufferMemoryInfo GetBindMemoryInfo() const;
};

}  // namespace gpu_resources