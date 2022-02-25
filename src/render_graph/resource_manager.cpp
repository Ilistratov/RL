#include "render_graph/resource_manager.h"

#include <cassert>

#include "base/base.h"

namespace render_graph {

void ResourceManager::AddBuffer(const std::string& name,
                                vk::DeviceSize size,
                                vk::MemoryPropertyFlags memory_flags) {
  assert(!buffers_.contains(name));
  buffers_[name] = gpu_resources::LogicalBuffer(size, memory_flags);
}

void ResourceManager::AddImage(const std::string& name,
                               vk::Extent2D extent,
                               vk::Format format,
                               vk::MemoryPropertyFlags memory_flags) {
  assert(!images_.contains(name));
  images_[name] = gpu_resources::LogicalImage(extent, format, memory_flags);
}

void ResourceManager::InitResources() {
  for (auto& [name, buffer] : buffers_) {
    buffer.Create();
    buffer.SetDebugName(name);
    buffer.RequestMemory(allocator_);
  }
  for (auto& [name, image] : images_) {
    image.Create();
    image.SetDebugName(name);
    image.RequestMemory(allocator_);
  }
  allocator_.Allocate();

  std::vector<vk::BindBufferMemoryInfo> buffer_bind_infos;
  buffer_bind_infos.reserve(buffers_.size());
  for (auto& [name, buffer] : buffers_) {
    buffer_bind_infos.push_back(buffer.GetBindMemoryInfo());
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.bindBufferMemory2(buffer_bind_infos);

  std::vector<vk::BindImageMemoryInfo> image_bind_infos;
  image_bind_infos.reserve(images_.size());
  for (auto& [name, image] : images_) {
    image_bind_infos.push_back(image.GetBindMemoryInfo());
  }
  device.bindImageMemory2(image_bind_infos);
}

void ResourceManager::RecordInitBarriers(vk::CommandBuffer cmd) const {
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers;
  image_barriers.reserve(images_.size());
  for (auto& [name, image] : images_) {
    image_barriers.push_back(image.GetInitBarrier());
  }
  vk::DependencyInfoKHR dep_info({}, {}, {}, image_barriers);
  cmd.pipelineBarrier2KHR(dep_info);
}

gpu_resources::LogicalBuffer& ResourceManager::GetBuffer(
    const std::string& name) {
  assert(buffers_.contains(name));
  return buffers_.at(name);
}

gpu_resources::LogicalImage& ResourceManager::GetImage(
    const std::string& name) {
  assert(images_.contains(name));
  return images_.at(name);
}

}  // namespace render_graph
