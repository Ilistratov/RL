#pragma once

#include <stdint.h>
#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>
#include <vulkan/vulkan_handles.hpp>

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
  void RequireProperties(ImageProperties properties);
  void DeclareAccess(ResourceAccess access, uint32_t pass_idx) const;

  vk::ImageView GetImageView() const noexcept;
  void CreateImageView();

  PhysicalImage* GetImage();
};

}  // namespace gpu_resources
