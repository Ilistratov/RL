#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/device_memory_allocator.h"
#include "gpu_resources/managers_common.h"

namespace gpu_resources {

struct BufferUsage {
  vk::AccessFlags2KHR access;
  vk::PipelineStageFlags2KHR stage;
  vk::BufferUsageFlags usage;

  BufferUsage& operator|=(BufferUsage other);
  bool IsModify() const;
  bool IsDependencyNeeded(BufferUsage other) const;
};

class BufferManager : public gpu_resources::ResourceManagerBase<BufferUsage> {
  Buffer buffer_;
  vk::DeviceSize size_;
  vk::MemoryPropertyFlags memory_properties_;

  vk::BufferUsageFlags GetAccumulatedUsage() const;

 public:
  BufferManager(vk::DeviceSize size, vk::MemoryPropertyFlags memory_properties);

  BufferManager(const BufferManager&) = delete;
  void operator=(const BufferManager&) = delete;

  void CreateBuffer();
  void ReserveMemoryBlock(DeviceMemoryAllocator& allocator) const;
  vk::BindBufferMemoryInfo GetBindMemoryInfo(
      DeviceMemoryAllocator& allocator) const;

  std::map<uint32_t, vk::BufferMemoryBarrier2KHR> GetBarriers() const;

  Buffer* GetBuffer();
};

}  // namespace gpu_resources
