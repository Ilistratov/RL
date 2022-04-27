#pragma once

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"

namespace gpu_resources {

class Image {
  AccessSyncManager access_manager_;
  vk::Image image_ = {};
  vk::Extent2D extent_ = {0, 0};
  vk::Format format_ = vk::Format::eUndefined;
  vk::ImageView image_view_ = {};
  vk::MemoryPropertyFlags memory_flags_;
  vk::ImageUsageFlags usage_flags_;
  MemoryBlock* memory_ = nullptr;

  friend class ResourceManager;

  Image(vk::Extent2D extent,
        vk::Format format,
        vk::MemoryPropertyFlags memory_flags);

 public:
  void AddUsage(uint32_t user_ind,
                ResourceUsage usage,
                vk::ImageUsageFlags usage_flags);

 private:
  void CreateVkImage();
  void SetDebugName(const std::string& debug_name) const;
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindImageMemoryInfo GetBindMemoryInfo() const;

  vk::ImageAspectFlags GetAspectFlags() const;

 public:
  Image() = default;

  Image(const Image&) = delete;
  void operator=(const Image&) = delete;

  Image(Image&& other) noexcept;
  void operator=(Image&& other) noexcept;

  void Swap(Image& other) noexcept;

  ~Image();

  Image(vk::Image image, vk::Extent2D extent, vk::Format format);
  vk::Image Release();

  vk::Image GetImage() const;
  vk::Extent2D GetExtent() const;
  vk::Format GetFormat() const;

  vk::ImageSubresourceRange GetSubresourceRange() const;
  vk::ImageSubresourceLayers GetSubresourceLayers() const;

  vk::ImageMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags,
      vk::ImageLayout src_layout = vk::ImageLayout::eUndefined,
      vk::ImageLayout dst_layout = vk::ImageLayout::eUndefined) const;
  vk::ImageMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);
  vk::ImageMemoryBarrier2KHR GetInitBarrier() const;

  void CreateImageView();
  vk::ImageView GetImageView() const;

  static void RecordBlit(vk::CommandBuffer cmd,
                         const Image& src,
                         const Image& dst);
};

}  // namespace gpu_resources
