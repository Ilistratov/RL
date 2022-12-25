#pragma once

#include <stdint.h>
#include <vulkan/vulkan.hpp>
#include "gpu_resources/device_memory_allocator.h"

namespace gpu_resources {

struct ImageProperties {
  vk::Extent2D extent = {0, 0};
  vk::Format format = vk::Format::eUndefined;
  vk::MemoryPropertyFlags memory_flags = {};
  vk::ImageUsageFlags usage_flags = {};

  static ImageProperties Unite(const ImageProperties& lhs,
                               const ImageProperties& rhs);
};

class PhysicalImage {
  uint32_t resource_idx_ = 0;
  vk::Image image_ = {};
  vk::ImageView image_view_ = {};
  ImageProperties properties_ = {};
  MemoryBlock* memory_ = nullptr;

  friend class ResourceManager;

  void CreateVkImage();
  void SetDebugName(const std::string& debug_name) const;
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindImageMemoryInfo GetBindMemoryInfo() const;

  vk::ImageAspectFlags GetAspectFlags() const;

 public:
  PhysicalImage() = default;
  PhysicalImage(uint32_t resource_idx, ImageProperties properties);
  /* Constructor for temporary unmanaged image so that we can use all the 'neat'
   * Features of the framework with vk::Image objects, that came from outside
   */
  PhysicalImage(vk::Image image, vk::Extent2D extent, vk::Format format);

  PhysicalImage(const PhysicalImage&) = delete;
  void operator=(const PhysicalImage&) = delete;

  PhysicalImage(PhysicalImage&& other) noexcept;
  void operator=(PhysicalImage&& other) noexcept;

  void Swap(PhysicalImage& other) noexcept;

  ~PhysicalImage();

  vk::Image Release();

  uint32_t GetIdx() const;
  vk::Image GetImage() const;
  vk::Extent2D GetExtent() const;
  vk::Format GetFormat() const;

  vk::ImageSubresourceRange GetSubresourceRange() const;
  vk::ImageSubresourceLayers GetSubresourceLayers() const;

  vk::ImageMemoryBarrier2KHR GenerateBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags,
      vk::ImageLayout src_layout = vk::ImageLayout::eUndefined,
      vk::ImageLayout dst_layout = vk::ImageLayout::eUndefined) const;

  void CreateImageView();
  vk::ImageView GetImageView() const;

  static void RecordBlit(vk::CommandBuffer cmd,
                         const PhysicalImage& src,
                         const PhysicalImage& dst);
};

}  // namespace gpu_resources
