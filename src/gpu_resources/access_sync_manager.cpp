#include "gpu_resources/access_sync_manager.h"

#include "utill/error_handling.h"

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
  DCHECK(layout == other.layout) << "Usage layouts incompatible";
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

void AccessSyncManager::AddUsage(uint32_t user_ind, ResourceUsage usage) {
  DCHECK(access_sequence_.empty() ||
         access_sequence_.back().user_ind < user_ind)
      << "Usages must be added in order of user_ind ascendance";
  if (access_sequence_.empty() ||
      IsDepNeeded(access_sequence_.back().usage, usage)) {
    access_sequence_.push_back({usage, user_ind});
  } else {
    access_sequence_.back().usage |= usage;
    access_sequence_.back().user_ind = user_ind;
  }
}

ResourceUsage AccessSyncManager::GetFirstUsage() const {
  DCHECK(!access_sequence_.empty());
  return access_sequence_[0].usage;
}

AccessDependency AccessSyncManager::GetUserDeps(uint32_t user_ind) {
  DCHECK(!access_sequence_.empty());
  DCHECK(next_dep_ind_ < access_sequence_.size());
  if (access_sequence_[next_dep_ind_].user_ind > user_ind) {
    return {};
  }
  DCHECK(access_sequence_[next_dep_ind_].user_ind == user_ind)
      << "Call sequence does not match |access_sequence_|";
  ResourceUsage src_usage = access_sequence_[next_dep_ind_].usage;
  ++next_dep_ind_;
  if (next_dep_ind_ == access_sequence_.size()) {
    next_dep_ind_ = 0;
  }
  ResourceUsage dst_usage = access_sequence_[next_dep_ind_].usage;
  return {src_usage, dst_usage};
}

}  // namespace gpu_resources
