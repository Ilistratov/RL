#include "mp_primitives/scan.h"

#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "gpu_resources/buffer.h"
#include "utill/error_handling.h"

namespace mp_primitives {

ScanPass::ScanPass(const shader::Loader& aggregate_shader,
                   const shader::Loader& scatter_shader,
                   pipeline_handler::DescriptorSet* d_set,
                   uint32_t n_elements,
                   gpu_resources::Buffer* values,
                   gpu_resources::Buffer* head_flags)
    : values_(values), head_flags_(head_flags) {
  stage_info_.stage_spacing = 1;
  stage_info_.n_elements = n_elements;
  stage_info_.is_segmented = head_flags_ ? 1 : 0;

  gpu_resources::BufferProperties requeired_buffer_propertires{};
  requeired_buffer_propertires.size = n_elements * sizeof(int);
  values_->RequireProperties(requeired_buffer_propertires);
  if (head_flags_) {
    head_flags_->RequireProperties(requeired_buffer_propertires);
  }

  if (head_flags_) {
    d_set->BulkBind(std::vector<gpu_resources::Buffer*>{values_, head_flags_},
                    true);
  } else {
    // so that second binding is also valid. Should reserach and start using
    // VK_EXT_descriptor_indexing to avoid this mess in the future
    d_set->BulkBind(std::vector<gpu_resources::Buffer*>{values_, values_},
                    true);
  }
  stage_info_pc_ = aggregate_shader.GeneratePushConstantRanges()[0];

  aggregate_pipeline_ = pipeline_handler::Compute(aggregate_shader, {d_set});
  scatter_pipeline_ = pipeline_handler::Compute(scatter_shader, {d_set});
}

void ScanPass::OnPreRecord() {
  gpu_resources::ResourceAccess access{};
  access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageRead |
                        vk::AccessFlagBits2KHR::eShaderStorageWrite;
  access.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  values_->DeclareAccess(access, GetPassIdx());
  if (head_flags_) {
    head_flags_->DeclareAccess(access, GetPassIdx());
  }
}

static inline vk::BufferMemoryBarrier2KHR GenerateScanStageBattier(
    gpu_resources::Buffer* buffer) {
  DCHECK(buffer);
  return buffer->GetBuffer()->GenerateBarrier(
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite |
          vk::AccessFlagBits2KHR::eShaderStorageRead,
      vk::PipelineStageFlagBits2KHR::eComputeShader,
      vk::AccessFlagBits2KHR::eShaderStorageWrite |
          vk::AccessFlagBits2KHR::eShaderStorageRead);
}

void ScanPass::OnRecord(vk::CommandBuffer primary_cmd,
                        const std::vector<vk::CommandBuffer>&) noexcept {
  const uint32_t elements_per_thread = 8;
  const uint32_t threads_per_group = 32;
  const uint32_t elements_per_group = elements_per_thread * threads_per_group;
  const std::vector<vk::BufferMemoryBarrier2KHR> barriers =
      head_flags_ ? std::vector{GenerateScanStageBattier(values_),
                                GenerateScanStageBattier(head_flags_)}
                  : std::vector{GenerateScanStageBattier(values_)};

  uint32_t head_count =
      (stage_info_.n_elements + elements_per_group - 1) / elements_per_group;
  while (head_count > 0) {
    primary_cmd.pushConstants(aggregate_pipeline_.GetLayout(),
                              stage_info_pc_.stageFlags, stage_info_pc_.offset,
                              stage_info_pc_.size, &stage_info_);
    aggregate_pipeline_.RecordDispatch(primary_cmd, head_count, 1, 1);

    primary_cmd.pipelineBarrier2KHR(vk::DependencyInfoKHR({}, {}, barriers));
    if (head_count == 1) {
      break;
    }
    stage_info_.stage_spacing *= elements_per_group;
    head_count = (head_count + elements_per_group - 1) / elements_per_group;
  }
  uint32_t head_spacing = stage_info_.stage_spacing;
  stage_info_.stage_spacing /= elements_per_group;
  head_count = (stage_info_.n_elements + head_spacing - 1) / head_spacing;
  while (stage_info_.stage_spacing > 0) {
    primary_cmd.pushConstants(scatter_pipeline_.GetLayout(),
                              stage_info_pc_.stageFlags, stage_info_pc_.offset,
                              stage_info_pc_.size, &stage_info_);
    scatter_pipeline_.RecordDispatch(primary_cmd, head_count - 1, 1, 1);
    head_spacing = stage_info_.stage_spacing;
    stage_info_.stage_spacing /= elements_per_group;
    head_count = (stage_info_.n_elements + head_spacing - 1) / head_spacing;
    if (stage_info_.stage_spacing > 0) {
      primary_cmd.pipelineBarrier2KHR(vk::DependencyInfoKHR({}, {}, barriers));
    }
  }
}

void ScanPass::Apply(render_graph::RenderGraph& render_graph,
                     ScanPass& dst_pass,
                     uint32_t n_elements,
                     gpu_resources::Buffer* values,
                     gpu_resources::Buffer* head_falgs) {
  shader::Loader aggregate_shader("shaders/mp-primitives/scan/aggregate.spv");
  shader::Loader scatter_shader("shaders/mp-primitives/scan/scatter.spv");
  pipeline_handler::DescriptorSet* d_set =
      aggregate_shader.GenerateDescriptorSet(render_graph.GetDescriptorPool(),
                                             0);
  dst_pass = ScanPass(aggregate_shader, scatter_shader, d_set, n_elements,
                      values, head_falgs);
  render_graph.AddPass(&dst_pass, vk::PipelineStageFlagBits2::eComputeShader);
}

}  // namespace mp_primitives
