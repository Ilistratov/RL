#pragma once

#include <map>
#include <string>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/logical_buffer.h"
#include "gpu_resources/logical_image.h"

namespace render_graph {

class ResourceManager {
  gpu_resources::DeviceMemoryAllocator allocator_;
  std::map<std::string, gpu_resources::LogicalBuffer> buffers_;
  std::map<std::string, gpu_resources::LogicalImage> images_;

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

  gpu_resources::LogicalBuffer& GetBuffer(const std::string& name);
  gpu_resources::LogicalImage& GetImage(const std::string& name);
};

}  // namespace render_graph
