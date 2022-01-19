#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/image.h"
#include "gpu_resources/managers_common.h"

namespace gpu_resources {

struct ImageUsage {
  vk::AccessFlags2KHR access;
  vk::PipelineStageFlags2KHR stage;
  vk::ImageUsageFlags usage;
  vk::ImageLayout layout;

  ImageUsage& operator|=(ImageUsage other);
  bool IsModify() const;
  bool IsDependencyNeeded(ImageUsage other) const;
};

class ImageManager : public gpu_resources::ResourceManagerBase<ImageUsage> {
  Image image_;
  vk::Extent2D extent_;
  vk::Format format_;
  vk::MemoryPropertyFlags memory_properties_;

  vk::ImageUsageFlags GetAccumulatedUsage() const;

 public:
  ImageManager(vk::Extent2D extent,
               vk::Format format,
               vk::MemoryPropertyFlags memory_properties);

  ImageManager(const ImageManager&) = delete;
  void operator=(const ImageManager&) = delete;

  void CreateImage();
  void ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const;
  vk::BindImageMemoryInfo GetBindMemoryInfo(
      DeviceMemoryAllocator& allocator) const;

  std::map<uint32_t, vk::ImageMemoryBarrier2KHR> GetBarriers() const;

  Image* GetImage();
};

}  // namespace gpu_resources
