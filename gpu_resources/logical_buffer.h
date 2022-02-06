#pragma once

#include "gpu_resources/access_sync_manager.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/resource_manager.h"

namespace gpu_resources {

class LogicalBuffer {
  AccessSyncManager access_manager_;
  std::vector<AccessDependency> dependencies_;
  ResourceUsage previous_usage_;
  uint32_t current_usage_ind_ = 0;

  vk::DeviceSize required_size_;
  vk::MemoryPropertyFlags required_memory_properties_;
  vk::BufferUsageFlags required_usage_flags_;

  ResourceManager* resource_manager_ = nullptr;
  PhysicalBuffer* bound_buffer_ = nullptr;
  bool is_persistent_ = false;

 public:
  LogicalBuffer(ResourceManager* resource_manager_,
                vk::DeviceSize required_size,
                vk::MemoryPropertyFlags required_memory_properties,
                bool is_persistent_);

  void AddUsage(uint32_t user_ind,
                vk::BufferUsageFlags usage_flags,
                vk::AccessFlags2KHR access,
                vk::PipelineStageFlags2KHR stage_flags);

  uint32_t GetFirstUserInd() const;
  uint32_t GetLastUserInd() const;
  PhysicalBuffer* GetBoundBuffer();

  // find suitable PhysicalBuffer in ResourceManager and bind it
  void OnInitBindBuffer();
  // return buffer with usage information to the resource manager so other
  // logical buffers can bind it.
  void OnInitRecycleBuffer();

  // if user_ind == first_user_ind_ than we might want to insert barrier prior
  // to Pass execution in order to wait for previous usage to complete. Returns
  // empty barrier if not needed. Assumed to be called in ascending order of
  // user_ind, starting from |GetFirstUserInd()| and ending at
  // |GetFirstUserInd()|, than returning back to |GetFirstUserInd()|
  vk::BufferMemoryBarrier2KHR GetPrePassBarrier(uint32_t user_ind);

  // GetPostPassBarrier - returns Barrier that current user needs to insert
  // after it's commands in order to sync his acces with the following passes
  // Returns empty barrier if not needed. Assumed to be called in ascending
  // order of user_ind, starting from |GetFirstUserInd()| and ending at
  // |GetFirstUserInd()|, than returning back to |GetFirstUserInd()|
  vk::BufferMemoryBarrier2KHR GetPostPassBarrier(uint32_t user_ind);
};

}  // namespace gpu_resources
