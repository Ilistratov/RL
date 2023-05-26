#include "mp_primitives/copy.h"
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "utill/error_handling.h"

namespace mp_primitives {
namespace detail {

CopyPass::CopyPass(gpu_resources::Buffer* src,
                   gpu_resources::Buffer* dst,
                   vk::BufferCopy2 copy_region)
    : copy_region_(copy_region), src_(src), dst_(dst) {
  gpu_resources::BufferProperties src_prop{
      .usage_flags = vk::BufferUsageFlagBits::eTransferSrc};
  gpu_resources::BufferProperties dst_prop{
      .usage_flags = vk::BufferUsageFlagBits::eTransferDst};
  DCHECK(src_ != nullptr) << "Src must be initialized buffer";
  DCHECK(dst_ != nullptr) << "Dst must be initialized buffer";
  DCHECK(copy_region.size > 0) << "Copy region cant be empty";
  // Don't add size requirement here as if it is not satisfied by now, it is
  // Likely to be an earlier error in the requirement specification.
  src_->RequireProperties(src_prop);
  dst_->RequireProperties(dst_prop);
}

CopyPass::CopyPass(CopyPass&& other) noexcept {
  Swap(other);
}

void CopyPass::operator=(CopyPass&& other) noexcept {
  CopyPass tmp(std::move(other));
  Swap(tmp);
}

void CopyPass::Swap(CopyPass& other) noexcept {
  std::swap(src_, other.src_);
  std::swap(dst_, other.dst_);
  std::swap(copy_region_, other.copy_region_);
}

void CopyPass::OnPreRecord() {
  gpu_resources::ResourceAccess src_access{
      .stage_flags = vk::PipelineStageFlagBits2::eTransfer,
      .access_flags = vk::AccessFlagBits2KHR::eTransferRead,
  };
  gpu_resources::ResourceAccess dst_access{
      .stage_flags = vk::PipelineStageFlagBits2::eTransfer,
      .access_flags = vk::AccessFlagBits2KHR::eTransferWrite,
  };
  src_->DeclareAccess(src_access, GetPassIdx());
  dst_->DeclareAccess(dst_access, GetPassIdx());
}

void CopyPass::OnRecord(vk::CommandBuffer primary_cmd,
                        const std::vector<vk::CommandBuffer>&) {
  primary_cmd.copyBuffer2(vk::CopyBufferInfo2(static_cast<vk::Buffer>(*src_),
                                              static_cast<vk::Buffer>(*dst_),
                                              copy_region_));
}

void TinyCopy::Apply(render_graph::RenderGraph& render_graph,
                     gpu_resources::Buffer* src,
                     gpu_resources::Buffer* dst,
                     vk::BufferCopy2 copy_region) {
  pass_ = CopyPass(src, dst, copy_region);
  render_graph.AddPass(&pass_, vk::PipelineStageFlagBits2::eTransfer);
}

}  // namespace detail
}  // namespace mp_primitives
