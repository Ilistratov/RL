#pragma once

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/physical_buffer.h"

namespace gpu_resources {

class LogicalBuffer {
  PhysicalBuffer buffer_;
  AccessSyncManager access_manager_;
  std::vector<AccessDependency> dependencies_;

  vk::DeviceSize required_size_;
  vk::MemoryPropertyFlags required_memory_flags_;
  vk::BufferUsageFlags required_usage_flags_;
  MemoryBlock* requested_memory_ = nullptr;
  uint32_t next_dep_ind_ = 0;

 public:
  LogicalBuffer(vk::DeviceSize required_size,
                vk::MemoryPropertyFlags required_memory_flags);

  void AddUsage(uint32_t user_ind,
                vk::BufferUsageFlags usage_flags,
                vk::AccessFlags2KHR access_flags,
                vk::PipelineStageFlags2KHR stage_flags);

  void Create(std::string debug_name = {});
  void RequestMemory(DeviceMemoryAllocator& allocator);
  vk::BindBufferMemoryInfo GetBindMemoryInfo() const;

  // GetPostPassBarrier - returns Barrier that current user needs to insert
  // after it's commands in order to sync his acces with the following
  // passes Returns empty barrier if not needed. Assumed to be called in
  // ascending order of user_ind, starting from |GetFirstUserInd()| and
  // ending at |GetFirstUserInd()|, than returning back to
  // |GetFirstUserInd()|
  vk::BufferMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);
};

}  // namespace gpu_resources
