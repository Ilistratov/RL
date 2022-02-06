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
  ResourceUsage src_usage;
  ResourceUsage dst_usage;
  uint32_t user_id;
};

class AccessSyncManager {
  struct UsageEntry {
    ResourceUsage usage;
    uint32_t user_id;
  };
  std::vector<UsageEntry> access_sequence_;
  uint32_t first_user_id_;

 public:
  static bool IsDepNeeded(ResourceUsage src_usage, ResourceUsage dst_usage);
  void Clear();
  void AddUsage(uint32_t user_id, ResourceUsage usage);

  uint32_t GetFirstUserId() const;
  uint32_t GetLastUserId() const;
  ResourceUsage GetFirstAccess() const;
  ResourceUsage GetLastAccess() const;
  std::vector<AccessDependency> GetUserDeps() const;
};

}  // namespace gpu_resources
