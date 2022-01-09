#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/device_memory_allocator.h"

namespace gpu_resources {

class BufferManager {
 public:
  struct Usage {
    vk::AccessFlags2KHR access;
    vk::PipelineStageFlags2KHR stage;
    vk::BufferUsageFlags usage;

    Usage& operator|=(const Usage& other);
    bool IsModify() const;
  };

 private:
  Buffer buffer_;
  vk::DeviceSize size_;
  vk::MemoryPropertyFlags memory_properties_;
  std::map<uint32_t, Usage> usage_by_ind_;

  vk::BufferUsageFlags GetAccumulatedUsage() const;

  Usage GetDstUsage(uint32_t src_user_ind) const;
  Usage GetSrcUsage(uint32_t dst_usage_ind) const;

 public:
  BufferManager(vk::DeviceSize size, vk::MemoryPropertyFlags memory_properties);

  BufferManager(const BufferManager&) = delete;
  void operator=(const BufferManager&) = delete;

  // It is considered that accesses are perfomed sequentially in order of
  // user_ind increasments. Same order as order in which tasks in gpu_executer
  // are executed.
  void AddUsage(uint32_t user_ind, Usage usage);
  void CreateBuffer();
  void ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const;
  vk::BindBufferMemoryInfo GetBindMemoryInfo(
      DeviceMemoryAllocator& allocator) const;

  std::map<uint32_t, vk::BufferMemoryBarrier2KHR> GetBarriers() const;

  Buffer* GetBuffer();
};

}  // namespace gpu_resources
