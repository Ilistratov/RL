#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/image.h"

namespace gpu_resources {

class ImageManager {
 public:
  struct Usage {
    vk::AccessFlags2KHR access;
    vk::PipelineStageFlags2KHR stage;
    vk::ImageUsageFlags usage;
    vk::ImageLayout layout;

    Usage& operator|=(const Usage& other);
    bool IsModify() const;
  };

 private:
  Image image_;
  vk::Extent2D extent_;
  vk::Format format_;
  vk::MemoryPropertyFlags memory_properties_;
  std::map<uint32_t, Usage> usage_by_ind_;

  vk::ImageUsageFlags GetAccumulatedUsage() const;
  bool IsLayoutTransitionNeeded(uint32_t user_ind) const;

  std::map<uint32_t, Usage>::const_iterator LoopedNext(
      std::map<uint32_t, Usage>::const_iterator it) const;
  std::map<uint32_t, Usage>::const_iterator LoopedPrev(
      std::map<uint32_t, Usage>::const_iterator it) const;
  Usage GetDstUsage(uint32_t src_user_ind) const;
  Usage GetSrcUsage(uint32_t dst_user_ind) const;

 public:
  ImageManager(vk::Extent2D extent,
               vk::Format format,
               vk::MemoryPropertyFlags memory_properties);

  ImageManager(const ImageManager&) = delete;
  void operator=(const ImageManager&) = delete;

  void AddUsage(uint32_t user_ind, Usage usage);
  void CreateImage();

  void ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const;
  vk::BindImageMemoryInfo GetBindMemoryInfo(
      DeviceMemoryAllocator& allocator) const;

  std::map<uint32_t, vk::ImageMemoryBarrier2KHR> GetBarriers() const;

  Image* GetImage();
};

}  // namespace gpu_resources
