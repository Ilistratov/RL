#pragma once

#include <stdint.h>
#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>
#include <vulkan/vulkan_handles.hpp>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"

namespace gpu_resources {

class Image {
  PhysicalImage* image_ = nullptr;
  PassAccessSyncronizer* syncronizer_;
  ImageProperties required_properties_;

  friend class ResourceManager;

  Image(ImageProperties properties, PassAccessSyncronizer* syncronizer);

 public:
  void DeclareAccess(ResourceAccess access, uint32_t pass_idx) const;

  static void RecordBlit(vk::CommandBuffer cmd,
                         const Image& src,
                         const Image& dst);

  vk::ImageView GetImageView() const noexcept;
  void CreateImageView();
};

}  // namespace gpu_resources
