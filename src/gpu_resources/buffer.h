#pragma once

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"

namespace gpu_resources {

class Buffer {
  AccessSyncManager access_manager_;
  vk::Buffer buffer_ = {};
  vk::DeviceSize size_ = 0;
  vk::BufferUsageFlags usage_flags_ = {};
  vk::MemoryPropertyFlags memory_flags_;
  MemoryBlock* memory_ = nullptr;

  friend class ResourceManager;

  Buffer(vk::DeviceSize size, vk::MemoryPropertyFlags memory_flags);

 public:
  void AddUsage(uint32_t user_ind,
                ResourceUsage usage,
                vk::BufferUsageFlags buffer_usage_flags);

 private:
  void CreateVkBuffer();
  void SetDebugName(const std::string& debug_name) const;
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindBufferMemoryInfo GetBindMemoryInfo() const;

 public:
  Buffer() = default;

  Buffer(const Buffer&) = delete;
  void operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept;
  void operator=(Buffer&& other) noexcept;
  void Swap(Buffer& other) noexcept;

  ~Buffer();

  vk::Buffer GetBuffer() const;
  vk::DeviceSize GetSize() const;
  void* GetMappingStart() const;
  vk::MappedMemoryRange GetMappedMemoryRange() const;

  vk::BufferMemoryBarrier2KHR GetBarrier(
      vk::PipelineStageFlags2KHR src_stage_flags,
      vk::AccessFlags2KHR src_access_flags,
      vk::PipelineStageFlags2KHR dst_stage_flags,
      vk::AccessFlags2KHR dst_access_flags) const;
  vk::BufferMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);

  static void RecordCopy(vk::CommandBuffer cmd,
                         const Buffer& src,
                         const Buffer& dst,
                         vk::DeviceSize src_offset,
                         vk::DeviceSize dst_offset,
                         vk::DeviceSize size);
  static void RecordCopy(vk::CommandBuffer cmd,
                         const Buffer& src,
                         const Buffer& dst,
                         const std::vector<vk::BufferCopy>& copy_regions);

  vk::DeviceSize LoadDataFromPtr(void* data,
                                 vk::DeviceSize data_size,
                                 vk::DeviceSize dst_offset);

  template <typename T>
  vk::DeviceSize LoadDataFromVec(const std::vector<T>& data,
                                 vk::DeviceSize dst_offset);
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
