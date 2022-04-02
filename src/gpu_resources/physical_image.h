#pragma once

#include <string>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/memory_block.h"

namespace gpu_resources {

class PhysicalImage {
  vk::Image image_ = {};
  vk::Extent2D extent_ = {0, 0};
  vk::Format format_ = vk::Format::eUndefined;
  vk::ImageView image_view_;

  friend class LogicalImage;

  bool IsDepthImage() const;
  vk::ImageAspectFlags GetAspectFlags() const;

 public:
  static const vk::Format kDepthFormats[];
  static const uint32_t kDepthFormatsCount;

  PhysicalImage() = default;
  PhysicalImage(vk::Extent2D extent,
                vk::Format format,
                vk::ImageUsageFlags image_usage);
  PhysicalImage(vk::Image image, vk::Extent2D extent, vk::Format format);

  PhysicalImage(const PhysicalImage&) = delete;
  void operator=(const PhysicalImage&) = delete;

  PhysicalImage(PhysicalImage&& other) noexcept;
  void operator=(PhysicalImage&& other) noexcept;
  void Swap(PhysicalImage& other) noexcept;
  vk::Image Release();

  vk::Image GetImage() const;
  vk::Extent2D GetExtent() const;
  vk::Format GetFormat() const;
  vk::MemoryRequirements GetMemoryRequierments() const;
  vk::BindImageMemoryInfo GetBindMemoryInfo(MemoryBlock memory_block) const;

  vk::ImageSubresourceRange GetSubresourceRange() const;
  vk::ImageSubresourceLayers GetSubresourceLayers() const;

  vk::ImageMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags,
      vk::ImageLayout src_layout = vk::ImageLayout::eUndefined,
      vk::ImageLayout dst_layout = vk::ImageLayout::eUndefined) const;

  void SetDebugName(const std::string& debug_name) const;

  void CreateImageView();
  vk::ImageView GetImageView() const;

  static void RecordBlit(vk::CommandBuffer cmd,
                         const PhysicalImage& src,
                         const PhysicalImage& dst);

  static bool IsFormatFeatureCombinationSupported(
      vk::Format format,
      vk::ImageTiling tiling,
      vk::FormatFeatureFlags features);
  static vk::Format GetDepthImageFormat();

  ~PhysicalImage();
};

}  // namespace gpu_resources
