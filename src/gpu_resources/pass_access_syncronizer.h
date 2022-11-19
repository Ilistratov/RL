#pragma once

#include <stdint.h>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"

namespace gpu_resources {

class PassAccessSyncronizer {
  std::vector<ResourceAccessSyncronizer> resource_syncronizers_;
  std::vector<std::vector<vk::ImageMemoryBarrier2KHR>> pass_image_barriers_;
  std::vector<std::vector<vk::BufferMemoryBarrier2KHR>> pass_buffer_barriers_;

 public:
  PassAccessSyncronizer() = default;
  PassAccessSyncronizer(uint32_t resource_count, uint32_t pass_count);

  void AddAccess(PhysicalBuffer* buffer,
                 ResourceAccess access,
                 uint32_t pass_idx);
  void AddAccess(PhysicalImage* image,
                 ResourceAccess access,
                 uint32_t pass_idx);
  void RecordPassCommandBuffers(vk::CommandBuffer cmd, uint32_t pass_idx);
};

}  // namespace gpu_resources
