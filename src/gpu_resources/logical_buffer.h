#pragma once

#include <string>
#include <vector>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/physical_buffer.h"
#include "utill/error_handling.h"

namespace gpu_resources {

class LogicalBuffer {
  PhysicalBuffer buffer_;
  AccessSyncManager access_manager_;

  vk::DeviceSize size_;
  vk::MemoryPropertyFlags memory_flags_;
  vk::BufferUsageFlags usage_flags_;
  MemoryBlock* memory_ = nullptr;

 public:
  LogicalBuffer() = default;
  LogicalBuffer(vk::DeviceSize size, vk::MemoryPropertyFlags memory_flags);

  LogicalBuffer(const LogicalBuffer&) = delete;
  void operator=(const LogicalBuffer&) = delete;

  LogicalBuffer(LogicalBuffer&& other) noexcept;
  void operator=(LogicalBuffer&& other) noexcept;
  void Swap(LogicalBuffer& other) noexcept;

  void Create();
  void SetDebugName(const std::string& debug_name) const;
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindBufferMemoryInfo GetBindMemoryInfo() const;
  PhysicalBuffer& GetPhysicalBuffer();

  void AddUsage(uint32_t user_ind,
                ResourceUsage usage,
                vk::BufferUsageFlags buffer_usage_flags);
  // GetPostPassBarrier - returns Barrier that current user needs to insert
  // after it's commands in order to sync his acces with the following
  // passes Returns empty barrier if not needed. Assumed to be called in
  // ascending order of user_ind, starting from |GetFirstUserInd()| and
  // ending at |GetFirstUserInd()|, than returning back to
  // |GetFirstUserInd()|
  vk::BufferMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);

  void* GetMappingStart() const;
  vk::MappedMemoryRange GetMappedMemoryRange() const;

  template <typename T>
  vk::DeviceSize LoadDataFromVec(const std::vector<T>& data,
                                 vk::DeviceSize dst_offset);
};

template <typename T>
static size_t GetDataSize(const std::vector<T>& data) {
  return sizeof(T) * data.size();
}

template <typename T>
vk::DeviceSize LogicalBuffer::LoadDataFromVec(const std::vector<T>& data,
                                              vk::DeviceSize dst_offset) {
  size_t data_size = sizeof(T) * data.size();
  if (data_size == 0) {
    return dst_offset;
  }
  if (dst_offset + data_size > size_) {
    LOG << "Failed to load data from vec to buffer, not enough space";
    return -1;
  }
  DCHECK(memory_) << "Memory not bound";
  DCHECK(memory_->mapping_start) << "Memory not mapped";
  memcpy((char*)memory_->mapping_start + dst_offset, data.data(), data_size);
  dst_offset += data_size;
  return dst_offset;
}

}  // namespace gpu_resources
