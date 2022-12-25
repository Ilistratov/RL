#include "gpu_resources/pass_access_syncronizer.h"

#include <utility>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "gpu_resources/common.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

PassAccessSyncronizer::PassAccessSyncronizer(uint32_t resource_count,
                                             uint32_t pass_count)
    : resource_syncronizers_(resource_count),
      pass_image_barriers_(pass_count),
      pass_buffer_barriers_(pass_count) {}

void PassAccessSyncronizer::AddAccess(PhysicalBuffer* buffer,
                                      ResourceAccess access,
                                      uint32_t pass_idx) {
  DCHECK(buffer) << kErrResourceIsNull;
  uint32_t buffer_idx = buffer->GetIdx();
  DCHECK(buffer_idx < resource_syncronizers_.size()) << kErrInvalidResourceIdx;
  AccessDependency dep =
      resource_syncronizers_[buffer_idx].AddAccess(pass_idx, access);
  if (dep.src.access_flags == vk::AccessFlagBits2KHR::eNone &&
      dep.dst.access_flags == vk::AccessFlagBits2KHR::eNone) {
    return;
  }
  DCHECK(dep.src_pass_idx < pass_buffer_barriers_.size()) << kErrInvalidPassIdx;
  pass_buffer_barriers_[dep.src_pass_idx].push_back(
      buffer->GenerateBarrier(dep.src.stage_flags, dep.src.access_flags,
                              dep.dst.stage_flags, dep.dst.access_flags));
}

void PassAccessSyncronizer::AddAccess(PhysicalImage* image,
                                      ResourceAccess access,
                                      uint32_t pass_idx) {
  DCHECK(image) << kErrResourceIsNull;
  uint32_t image_idx = image->GetIdx();
  DCHECK(image_idx < resource_syncronizers_.size()) << kErrInvalidResourceIdx;
  AccessDependency dep =
      resource_syncronizers_[image_idx].AddAccess(pass_idx, access);
  if (dep.src.access_flags == vk::AccessFlagBits2KHR::eNone &&
      dep.dst.access_flags == vk::AccessFlagBits2KHR::eNone) {
    return;
  }
  DCHECK(dep.src_pass_idx < pass_buffer_barriers_.size()) << kErrInvalidPassIdx;
  pass_image_barriers_[dep.src_pass_idx].push_back(image->GenerateBarrier(
      dep.src.stage_flags, dep.src.access_flags, dep.dst.stage_flags,
      dep.dst.access_flags, dep.src.layout, dep.dst.layout));
}

void PassAccessSyncronizer::RecordPassCommandBuffers(vk::CommandBuffer cmd,
                                                     uint32_t pass_idx) {
  DCHECK(pass_idx < pass_buffer_barriers_.size()) << kErrInvalidPassIdx;
  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers =
      std::move(pass_buffer_barriers_[pass_idx]);
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers =
      std::move(pass_image_barriers_[pass_idx]);
  cmd.pipelineBarrier2KHR(
      vk::DependencyInfoKHR({}, {}, buffer_barriers, image_barriers));
}

std::vector<vk::BufferMemoryBarrier2KHR>
PassAccessSyncronizer::GetBufferPostPassBarriers(uint32_t pass_idx) {
  std::vector<vk::BufferMemoryBarrier2KHR> res;
  pass_buffer_barriers_[pass_idx].swap(res);
  return res;
}
std::vector<vk::ImageMemoryBarrier2KHR>
PassAccessSyncronizer::GetImagePostPassBarriers(uint32_t pass_idx) {
  std::vector<vk::ImageMemoryBarrier2KHR> res;
  pass_image_barriers_[pass_idx].swap(res);
  return res;
}

}  // namespace gpu_resources
