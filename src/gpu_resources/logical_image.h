#pragma once

#include <string>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/physical_image.h"

namespace gpu_resources {

class LogicalImage {
  PhysicalImage image_;
  AccessSyncManager access_manager_;

  vk::Extent2D extent_;
  vk::Format format_;
  vk::MemoryPropertyFlags memory_flags_;
  vk::ImageUsageFlags usage_flags_;
  MemoryBlock* memory_ = nullptr;

 public:
  LogicalImage() = default;
  LogicalImage(vk::Extent2D extent,
               vk::Format format,
               vk::MemoryPropertyFlags memory_flags);

  LogicalImage(LogicalImage&& other) noexcept;
  void operator=(LogicalImage&& other) noexcept;

  void Swap(LogicalImage& other) noexcept;

  static LogicalImage CreateStorageImage(vk::Extent2D extent = {0, 0});

  LogicalImage(const LogicalImage&) = delete;
  void operator=(const LogicalImage&) = delete;

  void Create();
  void SetDebugName(const std::string& debug_name) const;
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindImageMemoryInfo GetBindMemoryInfo() const;
  PhysicalImage& GetPhysicalImage();

  void AddUsage(uint32_t user_ind,
                ResourceUsage usage,
                vk::ImageUsageFlags image_usage_flags);
  vk::ImageMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);
};

}  // namespace gpu_resources
