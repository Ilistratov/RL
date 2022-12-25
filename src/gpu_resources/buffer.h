#pragma once

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/resource_access_syncronizer.h"

namespace gpu_resources {

class Buffer {
  PhysicalBuffer* buffer_;
  PassAccessSyncronizer* syncronizer_;
  BufferProperties required_properties_;
  friend class ResourceManager;

  Buffer(BufferProperties properties, PassAccessSyncronizer* syncronizer);

 public:
  void RequireProperties(BufferProperties properties);
  void DeclareAccess(ResourceAccess access, uint32_t pass_idx) const;

  static void RecordCopy(vk::CommandBuffer cmd,
                         const Buffer& src,
                         const Buffer& dst,
                         vk::DeviceSize src_offset,
                         vk::DeviceSize dst_offset,
                         vk::DeviceSize size);
  static void RecordCopy(vk::CommandBuffer cmd,
                         const Buffer& src,
                         const Buffer& dst,
                         const std::vector<vk::BufferCopy2KHR>& copy_regions);

  vk::DeviceSize LoadDataFromPtr(void* data,
                                 vk::DeviceSize data_size,
                                 vk::DeviceSize dst_offset);

  template <typename T>
  vk::DeviceSize LoadDataFromVec(const std::vector<T>& data,
                                 vk::DeviceSize dst_offset);

  vk::Buffer GetVkBuffer() const noexcept;
  PhysicalBuffer* GetBuffer() const noexcept;
  vk::DeviceSize GetSize() const noexcept;
};

template <typename T>
static size_t GetDataSize(const std::vector<T>& data) {
  return sizeof(T) * data.size();
}

template <typename T>
vk::DeviceSize Buffer::LoadDataFromVec(const std::vector<T>& data,
                                       vk::DeviceSize dst_offset) {
  return LoadDataFromPtr((void*)data.data(), data.size() * sizeof(T),
                         dst_offset);
}

}  // namespace gpu_resources
