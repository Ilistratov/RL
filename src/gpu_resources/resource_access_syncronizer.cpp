#include "gpu_resources/resource_access_syncronizer.h"

#include "gpu_resources/common.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

bool ResourceAccess::IsModify() const {
  return (access_flags &
          (vk::AccessFlagBits2KHR::eAccelerationStructureWriteKHR |
           vk::AccessFlagBits2KHR::eColorAttachmentWrite |
           vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite |
           vk::AccessFlagBits2KHR::eHostWrite |
           vk::AccessFlagBits2KHR::eMemoryWrite |
           vk::AccessFlagBits2KHR::eShaderStorageWrite |
           vk::AccessFlagBits2KHR::eShaderWrite |
           vk::AccessFlagBits2KHR::eTransferWrite)) !=
         vk::AccessFlagBits2KHR::eNone;
}

ResourceAccess& ResourceAccess::operator|=(const ResourceAccess& other) {
  DCHECK(layout == other.layout) << kErrLayoutsIncompatible;
  stage_flags |= other.stage_flags;
  access_flags |= other.access_flags;
  return *this;
}

bool ResourceAccessSyncronizer::IsDepNeeded(ResourceAccess src_access,
                                            ResourceAccess dst_access) {
  return src_access.IsModify() || dst_access.IsModify() ||
         src_access.layout != dst_access.layout;
}

AccessDependency ResourceAccessSyncronizer::AddAccess(uint32_t pass_idx,
                                                      ResourceAccess access) {
  AccessDependency result = {};
  if (IsDepNeeded(current_unflushed_access_.access, access)) {
    result = AccessDependency{current_unflushed_access_.access, access,
                              current_unflushed_access_.pass_idx};
    if (result.src.layout != result.dst.layout &&
        result.dst.layout == vk::ImageLayout::eUndefined) {
      access.layout = result.dst.layout = result.src.layout;
    }
    current_unflushed_access_.access = access;
  } else {
    current_unflushed_access_.access |= access;
  }
  current_unflushed_access_.pass_idx = pass_idx;
  return result;
}

}  // namespace gpu_resources
