#pragma once

#include <string>

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/physical_buffer.h"

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
};

}  // namespace gpu_resources
