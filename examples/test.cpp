#include "test.h"

#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "mp_primitives/scan.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"
#include "shader/loader.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace examples {

// const uint32_t kNVals = 256 * 256 * 256 * 2;
const uint32_t kNVals = 512 * 512 * 64;
std::vector<int> g_values;
// std::vector<uint32_t> g_head_flags;

LoadToGpuPass::LoadToGpuPass(gpu_resources::Buffer* staging,
                             gpu_resources::Buffer* values)
    : staging_(staging), values_(values) {
  gpu_resources::BufferProperties required_transfer_src_properties{};
  required_transfer_src_properties.allocation_flags =
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT |
      VmaAllocationCreateFlagBits::
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  required_transfer_src_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc;
  required_transfer_src_properties.size = kNVals * sizeof(g_values[0]);
  staging_->RequireProperties(required_transfer_src_properties);
  gpu_resources::BufferProperties required_transfer_dst_properties{};
  required_transfer_dst_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferDst;
  required_transfer_dst_properties.size = kNVals * sizeof(g_values[0]);
  values_->RequireProperties(required_transfer_dst_properties);
}

void LoadToGpuPass::OnResourcesInitialized() noexcept {
  vk::DeviceSize dst_offset = 0;
  dst_offset = staging_->LoadDataFromPtr(
      g_values.data(), kNVals * sizeof(g_values[0]), dst_offset);
  auto device = base::Base::Get().GetContext().GetDevice();
  device.flushMappedMemoryRanges(staging_->GetBuffer()->GetMappedMemoryRange());
}

void LoadToGpuPass::OnPreRecord() {
  if (staging_ == nullptr) {
    return;
  }
  vk::PipelineStageFlags2KHR pass_stage =
      vk::PipelineStageFlagBits2KHR::eTransfer;
  gpu_resources::ResourceAccess transfer_src_access{};
  transfer_src_access.access_flags = vk::AccessFlagBits2KHR::eTransferRead;
  transfer_src_access.stage_flags = pass_stage;
  staging_->DeclareAccess(transfer_src_access, GetPassIdx());

  gpu_resources::ResourceAccess transfer_dst_access{};
  transfer_dst_access.access_flags = vk::AccessFlagBits2KHR::eTransferWrite;
  transfer_dst_access.stage_flags = pass_stage;
  values_->DeclareAccess(transfer_dst_access, GetPassIdx());
}

void LoadToGpuPass::OnRecord(vk::CommandBuffer primary_cmd,
                             const std::vector<vk::CommandBuffer>&) {
  if (staging_ == nullptr) {
    return;
  }
  gpu_resources::Buffer::RecordCopy(primary_cmd, *staging_, *values_, 0, 0,
                                    kNVals * sizeof(g_values[0]));
  staging_ = nullptr;
}

LoadToCpuPass::LoadToCpuPass(gpu_resources::Buffer* staging,
                             gpu_resources::Buffer* values)
    : staging_(staging), values_(values) {
  gpu_resources::BufferProperties required_transfer_src_properties{};
  required_transfer_src_properties.allocation_flags =
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT |
      VmaAllocationCreateFlagBits::
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  required_transfer_src_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc;
  values_->RequireProperties(required_transfer_src_properties);
  required_transfer_src_properties.size = g_values.size() * sizeof(g_values[0]);
  gpu_resources::BufferProperties required_transfer_dst_properties{};
  required_transfer_dst_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferDst;
  required_transfer_dst_properties.size = g_values.size() * sizeof(g_values[0]);
  staging_->RequireProperties(required_transfer_dst_properties);
}

void LoadToCpuPass::OnPreRecord() {
  vk::PipelineStageFlags2KHR pass_stage =
      vk::PipelineStageFlagBits2KHR::eTransfer;
  gpu_resources::ResourceAccess access{};
  access.access_flags = vk::AccessFlagBits2KHR::eTransferRead;
  access.stage_flags = pass_stage;
  values_->DeclareAccess(access, GetPassIdx());

  access.access_flags = vk::AccessFlagBits2KHR::eTransferWrite;
  staging_->DeclareAccess(access, GetPassIdx());
}

void LoadToCpuPass::OnRecord(vk::CommandBuffer primary_cmd,
                             const std::vector<vk::CommandBuffer>&) {
  gpu_resources::Buffer::RecordCopy(primary_cmd, *values_, *staging_, 0, 0,
                                    g_values.size() * sizeof(g_values[0]));
}

TestRenderer::TestRenderer() {
  gpu_resources::Buffer* staging =
      render_graph_.GetResourceManager().AddBuffer({});
  gpu_resources::Buffer* values =
      render_graph_.GetResourceManager().AddBuffer({});
  gpu_resources::Buffer* pos = render_graph_.GetResourceManager().AddBuffer({});
  g_values = std::vector<int>(kNVals, 1);
  for (uint32_t i = 0; i < kNVals; i++) {
    g_values[i] = kNVals - i - 1;
  }
  pos->RequireProperties(gpu_resources::BufferProperties{
      .size = g_values.size() * sizeof(g_values[0])});
  load_to_gpu_ = LoadToGpuPass(staging, values);
  render_graph_.AddPass(&load_to_gpu_,
                        vk::PipelineStageFlagBits2KHR::eTransfer);
  sort_.Apply(render_graph_, g_values.size(), values, pos);
  load_to_cpu_ = LoadToCpuPass(staging, values);
  render_graph_.AddPass(&load_to_cpu_,
                        vk::PipelineStageFlagBits2KHR::eTransfer);
  render_graph_.Init();
  LOG << "computation start";
  render_graph_.RenderFrame();
  LOG << "submit done";
  base::Base::Get().GetContext().GetDevice().waitIdle();
  LOG << "computation end";
  base::Base::Get().GetContext().GetDevice().invalidateMappedMemoryRanges(
      staging->GetBuffer()->GetMappedMemoryRange());
  LOG << "cache invalidated";
  memcpy_s(g_values.data(), g_values.size() * sizeof(int),
           staging->GetBuffer()->GetMappingStart(),
           g_values.size() * sizeof(int));
  LOG << "memcpy done";
  DCHECK(std::is_sorted(g_values.begin(), g_values.end())) << "Not sorted";
  LOG << "check done";
}

}  // namespace examples
