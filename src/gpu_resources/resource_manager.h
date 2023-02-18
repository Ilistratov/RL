#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"

namespace gpu_resources {

class ResourceManager {
  PassAccessSyncronizer syncronizer_;
  std::list<Buffer> buffers_;
  std::list<Image> images_;
  std::vector<PhysicalBuffer> physical_buffers_;
  std::vector<PhysicalImage> physical_images_;

  uint32_t CreateAndMapPhysicalResources();
  void InitPhysicalResources();

 public:
  ResourceManager() = default;

  ResourceManager(const ResourceManager&) = delete;
  void operator=(const ResourceManager&) = delete;

  Buffer* AddBuffer(BufferProperties properties);
  Image* AddImage(ImageProperties properties);
  PassAccessSyncronizer* GetAccessSyncronizer();

  void InitResources(uint32_t pass_count);
};

}  // namespace gpu_resources
