#pragma once

#include <vulkan/vulkan.hpp>

#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class Image {
  vk::Image image_ = {};
  vk::Extent2D extent_ = {0, 0};
  vk::Format format_ = vk::Format::eUndefined;
  bool is_managed_ = true;

 public:
  Image() = default;
  Image(vk::Extent2D extent,
        vk::Format format,
        vk::ImageUsageFlags image_usage);
  Image(vk::Image image, vk::Extent2D extent, vk::Format format);

  Image(const Image&) = delete;
  void operator=(const Image&) = delete;

  Image(Image&& other) noexcept;
  void operator=(Image&& other) noexcept;
  void Swap(Image& other) noexcept;

  vk::Image GetImage() const;
  vk::Extent2D GetExtent() const;
  vk::Format GetFormat() const;
  bool IsManaged() const;

  vk::BindImageMemoryInfo GetBindMemoryInfo(MemoryBlock memory) const;
  vk::MemoryRequirements GetMemoryRequierments() const;

  vk::ImageSubresourceRange GetSubresourceRange() const;
  vk::ImageSubresourceLayers GetSubresourceLayers() const;

  vk::ImageMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags,
      vk::ImageLayout src_layout = vk::ImageLayout::eUndefined,
      vk::ImageLayout dst_layout = vk::ImageLayout::eUndefined) const;

  static void RecordBlit(vk::CommandBuffer cmd,
                         const Image& src,
                         const Image& dst);

  ~Image();
};

}  // namespace gpu_resources
