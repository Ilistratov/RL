#pragma once

#include <utility>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/physical_buffer.h"

namespace gpu_resources {

class ResourceManager {
 public:
  ResourceManager() = default;
  ResourceManager(const ResourceManager&) = delete;
  void operator=(const ResourceManager&) = delete;

  std::pair<PhysicalBuffer*, ResourceUsage> GetBuffer(
      vk::DeviceSize required_size,
      vk::MemoryPropertyFlags required_memory_properties,
      vk::BufferUsageFlags required_usage_flags);
  PhysicalBuffer* RecycleBuffer(PhysicalBuffer* buffer, ResourceUsage usage);
};

}  // namespace gpu_resources
