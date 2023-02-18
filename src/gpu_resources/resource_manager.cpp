#include "gpu_resources/resource_manager.h"

#include <stdint.h>
#include <string>

#include <vulkan/vulkan.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "gpu_resources/resource_access_syncronizer.h"

namespace gpu_resources {

Buffer* ResourceManager::AddBuffer(BufferProperties properties) {
  buffers_.push_back(Buffer(properties, &syncronizer_));
  return &buffers_.back();
}

Image* ResourceManager::AddImage(ImageProperties properties) {
  auto& swapchain = base::Base::Get().GetSwapchain();
  if (properties.extent.width == 0 || properties.extent.height == 0) {
    properties.extent = swapchain.GetExtent();
  }
  if (properties.format == vk::Format::eUndefined) {
    properties.format = swapchain.GetFormat();
  }
  images_.push_back(Image(properties, &syncronizer_));
  return &images_.back();
}

PassAccessSyncronizer* ResourceManager::GetAccessSyncronizer() {
  return &syncronizer_;
}

// Simple 1:1 mapping for now. Can be replaced when transient resources are
// supported
uint32_t ResourceManager::CreateAndMapPhysicalResources() {
  uint32_t resource_count = 0;
  for (auto& buffer : buffers_) {
    PhysicalBuffer physical_buffer(resource_count, buffer.required_properties_);
    resource_count += 1;
    physical_buffers_.push_back(std::move(physical_buffer));
  }

  for (auto& image : images_) {
    PhysicalImage physical_image(resource_count, image.required_properties_);
    resource_count += 1;
    physical_images_.push_back(std::move(physical_image));
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
    buffer.SetDebugName(std::string("rg-buffer-") + std::to_string(idx));
    idx += 1;
  }

  idx = 0;

  for (auto& image : physical_images_) {
    image.SetDebugName(std::string("rg-image-") + std::to_string(idx));
    idx += 1;
  }
}

void ResourceManager::InitResources(uint32_t pass_count) {
  uint32_t resource_count = CreateAndMapPhysicalResources();
  syncronizer_ = PassAccessSyncronizer(resource_count, pass_count);
  InitPhysicalResources();
  ResourceAccess initial_access;
  initial_access.layout = vk::ImageLayout::eUndefined;
  initial_access.stage_flags = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
  for (auto& image : physical_images_) {
    syncronizer_.AddAccess(&image, initial_access, pass_count);
  }
}

}  // namespace gpu_resources
