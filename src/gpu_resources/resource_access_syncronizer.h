#pragma once

#include <vulkan/vulkan.hpp>

namespace gpu_resources {

struct ResourceAccess {
  vk::PipelineStageFlags2KHR stage_flags = vk::PipelineStageFlagBits2KHR::eNone;
  vk::AccessFlags2KHR access_flags = vk::AccessFlagBits2KHR::eNone;
  vk::ImageLayout layout = vk::ImageLayout::eUndefined;

  bool IsModify() const;
  ResourceAccess& operator|=(const ResourceAccess& other);
};

struct AccessDependency {
  ResourceAccess src = {};
  ResourceAccess dst = {};
  uint32_t src_pass_idx = 0;
};

class ResourceAccessSyncronizer {
  struct AccessEntry {
    ResourceAccess access;
    uint32_t pass_idx;
  };
  AccessEntry current_unflushed_access_ = {};

 public:
  static bool IsDepNeeded(ResourceAccess src_access, ResourceAccess dst_access);
  AccessDependency AddAccess(uint32_t pass_idx, ResourceAccess access);
};

}  // namespace gpu_resources
