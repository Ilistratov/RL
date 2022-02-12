#pragma once

#include <vulkan/vulkan.hpp>

namespace gpu_resources {

struct ResourceUsage {
  vk::PipelineStageFlags2KHR stage = {};
  vk::AccessFlags2KHR access = {};
  vk::ImageLayout layout = vk::ImageLayout::eUndefined;

  bool IsModify() const;
  ResourceUsage& operator|=(const ResourceUsage& other);
};

struct AccessDependency {
  ResourceUsage src_usage = {};
  ResourceUsage dst_usage = {};
};

class AccessSyncManager {
  struct UsageEntry {
    ResourceUsage usage;
    uint32_t user_ind;
  };
  std::vector<UsageEntry> access_sequence_;
  uint32_t first_user_ind_;
  uint32_t next_dep_ind_ = 0;

 public:
  static bool IsDepNeeded(ResourceUsage src_usage, ResourceUsage dst_usage);
  void Clear();
  void AddUsage(uint32_t user_ind, ResourceUsage usage);
  uint32_t GetFirstUserInd() const;
  AccessDependency GetUserDeps(uint32_t user_ind);
};

}  // namespace gpu_resources
