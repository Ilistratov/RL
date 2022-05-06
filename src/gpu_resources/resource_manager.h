#pragma once

#include <map>
#include <string>

#include "gpu_resources/buffer.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/image.h"

namespace gpu_resources {

class ResourceManager {
  gpu_resources::DeviceMemoryAllocator allocator_;
  std::map<std::string, gpu_resources::Buffer> buffers_;
  std::map<std::string, gpu_resources::Image> images_;

 public:
  ResourceManager() = default;

  ResourceManager(const ResourceManager&) = delete;
  void operator=(const ResourceManager&) = delete;

  void AddBuffer(const std::string& name,
                 vk::DeviceSize size,
                 vk::MemoryPropertyFlags memory_flags);
  void AddImage(const std::string& name,
                vk::Extent2D extent,
                vk::Format format,
                vk::MemoryPropertyFlags memory_flags);
  void InitResources();
  void RecordInitBarriers(vk::CommandBuffer cmd) const;

  gpu_resources::Buffer& GetBuffer(const std::string& name);
  gpu_resources::Image& GetImage(const std::string& name);
};

}  // namespace gpu_resources
