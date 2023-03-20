#include "mp_primitives/scan.h"

#include <stdint.h>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/physical_buffer.h"
#include "pipeline_handler/descriptor_set.h"
#include "render_graph/pass.h"
#include "utill/error_handling.h"

namespace mp_primitives {
namespace detail {

const static uint32_t kScanNElementSize = sizeof(uint32_t);
const static uint32_t kScanNElementsPerThtead = 8;
const static uint32_t kScanNThreadsPerGroup = 32;
const static uint32_t kScanNElementsPerGroup =
    kScanNElementsPerThtead * kScanNThreadsPerGroup;

const static char* kScanAggregateShaderPath =
    "shaders/mp_primitives/scan/aggregate.spv";
const static char* kScanScatterShaderPath =
    "shaders/mp_primitives/scan/scatter.spv";

ScanStagePass::ScanStagePass(vk::PushConstantRange stage_info_pc,
                             pipeline_handler::Compute* pipeline,
                             uint32_t stage_spacing,
                             uint32_t n_elements,
                             uint32_t n_groups,
                             gpu_resources::Buffer* values,
                             gpu_resources::Buffer* head_flags)
    : stage_info_pc_(stage_info_pc),
      pipeline_(pipeline),
      values_(values),
      head_flags_(head_flags),
      stage_spacing_(stage_spacing),
      n_elements_(n_elements),
      n_groups_(n_groups) {}

ScanStagePass::ScanStagePass(ScanStagePass&& other) noexcept {
  Swap(other);
}

void ScanStagePass::operator=(ScanStagePass&& other) noexcept {
  ScanStagePass tmp(std::move(other));
  Swap(tmp);
}

void ScanStagePass::Swap(ScanStagePass& other) noexcept {
  render_graph::Pass::Swap(other);
  std::swap(stage_info_pc_, other.stage_info_pc_);
  std::swap(pipeline_, other.pipeline_);
  std::swap(values_, other.values_);
  std::swap(head_flags_, other.head_flags_);
  std::swap(stage_spacing_, other.stage_spacing_);
  std::swap(n_elements_, other.n_elements_);
  std::swap(n_groups_, other.n_groups_);
}

void ScanStagePass::OnPreRecord() {
  gpu_resources::ResourceAccess access{};
  access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageRead |
                        vk::AccessFlagBits2KHR::eShaderStorageWrite;
  access.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  values_->DeclareAccess(access, GetPassIdx());
  if (head_flags_) {
    head_flags_->DeclareAccess(access, GetPassIdx());
  }
}

void ScanStagePass::OnRecord(vk::CommandBuffer primary_cmd,
                             const std::vector<vk::CommandBuffer>&) {
  ScanStageInfo pc;
  pc.stage_spacing = stage_spacing_;
  pc.n_elements = n_elements_;
  pc.is_segmented = (head_flags_ == nullptr) ? 0 : 1;
  primary_cmd.pushConstants(pipeline_->GetLayout(), stage_info_pc_.stageFlags,
                            stage_info_pc_.offset, stage_info_pc_.size, &pc);
  pipeline_->RecordDispatch(primary_cmd, n_groups_, 1, 1);
}

void Scan::Apply(render_graph::RenderGraph& render_graph,
                 uint32_t n_elements,
                 gpu_resources::Buffer* values,
                 gpu_resources::Buffer* head_flags) {
  DCHECK(passes_.empty()) << "This instance is already bound";
  DCHECK(n_elements > 0) << "Cant scan empty array";
  DCHECK(values != nullptr) << "Must specify valid buffer to scan";
  shader::Loader aggregate_shader(kScanAggregateShaderPath);
  shader::Loader scatter_shader(kScanScatterShaderPath);
  vk::PushConstantRange pc_range =
      aggregate_shader.GeneratePushConstantRanges()[0];
  pipeline_handler::DescriptorSet* dset =
      aggregate_shader.GenerateDescriptorSet(render_graph.GetDescriptorPool(),
                                             0);
  gpu_resources::BufferProperties properties{
      .size = n_elements * kScanNElementSize,
      .usage_flags = vk::BufferUsageFlagBits::eStorageBuffer};
  values->RequireProperties(properties);
  if (head_flags) {
    head_flags->RequireProperties(properties);
    dset->BulkBind(std::vector<gpu_resources::Buffer*>{values, head_flags},
                   true);
  } else {
    // so that second binding is also valid. Should reserach and start using
    // VK_EXT_descriptor_indexing to avoid this mess in the future
    dset->BulkBind(std::vector<gpu_resources::Buffer*>{values, values}, true);
  }
  aggregate_pipeline_ = pipeline_handler::Compute(aggregate_shader, {dset});
  scatter_pipeline_ = pipeline_handler::Compute(scatter_shader, {dset});

  uint32_t head_count =
      (n_elements + kScanNElementsPerGroup - 1) / kScanNElementsPerGroup;
  uint32_t stage_spacing = 1;

  while (head_count > 0) {
    passes_.push_back(ScanStagePass(pc_range, &aggregate_pipeline_,
                                    stage_spacing, n_elements, head_count,
                                    values, head_flags));
    if (head_count == 1) {
      break;
    }
    stage_spacing *= kScanNElementsPerGroup;
    head_count =
        (head_count + kScanNElementsPerGroup - 1) / kScanNElementsPerGroup;
  }

  head_count = (n_elements + stage_spacing - 1) / stage_spacing;
  while (stage_spacing / kScanNElementsPerGroup > 0) {
    passes_.push_back(ScanStagePass(
        pc_range, &scatter_pipeline_, stage_spacing / kScanNElementsPerGroup,
        n_elements, head_count - 1, values, head_flags));
    stage_spacing /= kScanNElementsPerGroup;
    head_count = (n_elements + stage_spacing - 1) / stage_spacing;
  }

  for (auto& pass : passes_) {
    render_graph.AddPass(&pass, vk::PipelineStageFlagBits2KHR::eComputeShader);
  }
}

}  // namespace detail

}  // namespace mp_primitives
