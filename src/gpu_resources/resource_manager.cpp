#include "gpu_resources/resource_manager.h"

#include <stdint.h>
#include <string>

#include "base/base.h"

#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"

namespace gpu_resources {

Buffer* ResourceManager::AddBuffer(BufferProperties properties) {
  buffers_.push_back(Buffer(properties, &syncronizer_));
  return &buffers_.back();
}

Image* ResourceManager::AddImage(ImageProperties properties) {
  images_.push_back(Image(properties, &syncronizer_));
  return &images_.back();
}

// Simple 1:1 mapping for now. Can be replaced when transient resources are
// supported
uint32_t ResourceManager::CreateAndMapPhysicalResources() {
  uint32_t resource_count = 0;
  for (auto& buffer : buffers_) {
    PhysicalBuffer physical_buffer(resource_count, buffer.required_properties_);
    resource_count += 1;
    physical_buffers_.push_back(physical_buffer);
  }

  for (auto& image : images_) {
    PhysicalImage physical_image(resource_count, image.required_properties_);
    resource_count += 1;
    physical_images_.push_back(physical_image);
  }

  uint32_t idx = 0;
  for (auto& buffer : buffers_) {
    buffer.buffer_ = &physical_buffers_[idx];
    idx += 1;
  }

  idx = 0;
  for (auto& image : images_) {
    image.image_ = &physical_images_[idx];
    idx += 1;
  }
  return resource_count;
}

void ResourceManager::InitPhysicalResources() {
  uint32_t idx = 0;
  for (auto& buffer : physical_buffers_) {
    buffer.CreateVkBuffer();
    buffer.SetDebugName(std::string("rg-buffer-") + std::to_string(idx));
    idx += 1;
    buffer.RequestMemory(allocator_);
  }

  idx = 0;

  for (auto& image : physical_images_) {
    image.CreateVkImage();
    image.SetDebugName(std::string("rg-image-") + std::to_string(idx));
    idx += 1;
    image.RequestMemory(allocator_);
  }
}

void ResourceManager::BindPhysicalResourcesMemory() {
  std::vector<vk::BindBufferMemoryInfo> buffer_bind_infos;
  buffer_bind_infos.reserve(buffers_.size());
  for (auto& buffer : physical_buffers_) {
    buffer_bind_infos.push_back(buffer.GetBindMemoryInfo());
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  if (!buffer_bind_infos.empty()) {
    device.bindBufferMemory2(buffer_bind_infos);
  }

  std::vector<vk::BindImageMemoryInfo> image_bind_infos;
  image_bind_infos.reserve(images_.size());
  for (auto& image : physical_images_) {
    image_bind_infos.push_back(image.GetBindMemoryInfo());
  }
  if (!image_bind_infos.empty()) {
    device.bindImageMemory2(image_bind_infos);
  }
}

void ResourceManager::InitResources(uint32_t pass_count) {
  uint32_t resource_count = CreateAndMapPhysicalResources();
  syncronizer_ = PassAccessSyncronizer(resource_count, pass_count);
  InitPhysicalResources();
  allocator_.Allocate();
  BindPhysicalResourcesMemory();
}

}  // namespace gpu_resources
