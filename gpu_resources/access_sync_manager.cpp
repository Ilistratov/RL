#include "gpu_resources/access_sync_manager.h"

#include <cassert>

namespace gpu_resources {

bool ResourceUsage::IsModify() const {
  return (access | (vk::AccessFlagBits2KHR::eAccelerationStructureWrite |
                    vk::AccessFlagBits2KHR::eColorAttachmentWrite |
                    vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite |
                    vk::AccessFlagBits2KHR::eHostWrite |
                    vk::AccessFlagBits2KHR::eMemoryWrite |
                    vk::AccessFlagBits2KHR::eShaderStorageWrite |
                    vk::AccessFlagBits2KHR::eShaderWrite |
                    vk::AccessFlagBits2KHR::eTransferWrite)) !=
         vk::AccessFlagBits2KHR::eNone;
}

ResourceUsage& ResourceUsage::operator|=(const ResourceUsage& other) {
  assert(layout == other.layout);
  stage |= other.stage;
  access |= other.access;
  return *this;
}

bool AccessSyncManager::IsDepNeeded(ResourceUsage src_usage,
                                    ResourceUsage dst_usage) {
  return src_usage.IsModify() || dst_usage.IsModify() ||
         src_usage.layout != dst_usage.layout;
}

void AccessSyncManager::Clear() {
  access_sequence_.clear();
}

void AccessSyncManager::AddUsage(uint32_t user_id, ResourceUsage usage) {
  if (access_sequence_.empty() ||
      IsDepNeeded(access_sequence_.back().usage, usage)) {
    access_sequence_.push_back({usage, user_id});
  } else {
    access_sequence_.back().usage |= usage;
    access_sequence_.back().user_id = user_id;
  }
}

ResourceUsage AccessSyncManager::GetFirstAccess() const {
  assert(!access_sequence_.empty());
  return access_sequence_.front().usage;
}

ResourceUsage AccessSyncManager::GetLastAccess() const {
  assert(!access_sequence_.empty());
  return access_sequence_.back().usage;
}

std::vector<AccessDependency> AccessSyncManager::GetUserDeps() const {
  assert(!access_sequence_.empty());
  std::vector<AccessDependency> res(access_sequence_.size() - 1);
  for (uint32_t i = 0; i + 1 < access_sequence_.size(); i++) {
    res[i] = {access_sequence_[i].usage, access_sequence_[i + 1].usage,
              access_sequence_[i].user_id};
  }
  return res;
}

}  // namespace gpu_resources
