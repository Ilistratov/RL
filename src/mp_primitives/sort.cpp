#include "mp_primitives/sort.h"
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "gpu_resources/buffer.h"
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/resource_access_syncronizer.h"
#include "mp_primitives/scan.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_set.h"
#include "shader/loader.h"

namespace mp_primitives {
namespace detail {

struct StageInfoPC {
  uint32_t n_elements;
  uint32_t n_bit_offset;
  uint32_t n_groups;
};

// Should be coherent with shaders/mp-primitives/sort/common.hlsl
const uint32_t kNElementsPerThread = 8;
const uint32_t kNThreadsPerGroup = 64;
const uint32_t kNElementsPerGroup = kNThreadsPerGroup * kNElementsPerThread;
const uint32_t kNHistBuckets = (1 << kSortBitsPerPhase);

static inline uint32_t GroupsPerInvocation(uint32_t n_elements) {
  return (n_elements + kNElementsPerGroup) / kNElementsPerGroup;
}

static inline void RecordSortPhaseInvocation(
    vk::CommandBuffer cmd,
    pipeline_handler::Compute& pipeline,
    vk::PushConstantRange pc_range,
    uint32_t n_elements,
    uint32_t n_bit_offset) {
  StageInfoPC push_constants{.n_elements = n_elements,
                             .n_bit_offset = n_bit_offset,
                             .n_groups = GroupsPerInvocation(n_elements)};
  cmd.pushConstants(pipeline.GetLayout(), pc_range.stageFlags, pc_range.offset,
                    pc_range.size, &push_constants);
  pipeline.RecordDispatch(cmd, push_constants.n_groups, 1, 1);
}

SortBlockPresortPass::SortBlockPresortPass(
    const shader::Loader& presort_shader,
    pipeline_handler::DescriptorPool& dpool,
    pipeline_handler::DescriptorSet* key_hist_dset,
    uint32_t n_elements,
    uint32_t n_bit_offset,
    gpu_resources::Buffer* key_buffer,
    gpu_resources::Buffer* pos_buffer,
    gpu_resources::Buffer* key_hist_buffer)
    : pc_range_(presort_shader.GeneratePushConstantRanges()[0]),
      n_elements_(n_elements),
      n_bit_offset_(n_bit_offset),
      key_buffer_(key_buffer),
      pos_buffer_(pos_buffer),
      key_hist_buffer_(key_hist_buffer) {
  pipeline_handler::DescriptorSet* sort_data_dset =
      presort_shader.GenerateDescriptorSet(dpool, 0);
  sort_data_dset->BulkBind(
      std::vector<gpu_resources::Buffer*>{key_buffer_, pos_buffer_});
  pipeline_ = pipeline_handler::Compute(presort_shader,
                                        {sort_data_dset, key_hist_dset});

  gpu_resources::BufferProperties properties{
      .size = sizeof(uint32_t) * n_elements,
      .usage_flags = vk::BufferUsageFlagBits::eStorageBuffer};
  key_buffer_->RequireProperties(properties);
  pos_buffer_->RequireProperties(properties);
  properties.size = GroupsPerInvocation(n_elements_) * kNHistBuckets;
  key_hist_buffer_->RequireProperties(properties);
}

void SortBlockPresortPass::OnPreRecord() {
  gpu_resources::ResourceAccess compute_buffer_rw{};
  compute_buffer_rw.access_flags = vk::AccessFlagBits2KHR::eShaderStorageRead |
                                   vk::AccessFlagBits2KHR::eShaderStorageWrite;
  compute_buffer_rw.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  key_buffer_->DeclareAccess(compute_buffer_rw, GetPassIdx());
  pos_buffer_->DeclareAccess(compute_buffer_rw, GetPassIdx());
  key_hist_buffer_->DeclareAccess(compute_buffer_rw, GetPassIdx());
}

void SortBlockPresortPass::OnRecord(vk::CommandBuffer primary_cmd,
                                    const std::vector<vk::CommandBuffer>&) {
  RecordSortPhaseInvocation(primary_cmd, pipeline_, pc_range_, n_elements_,
                            n_bit_offset_);
}

SortScatterPass::SortScatterPass(const shader::Loader& scatter_shader,
                                 pipeline_handler::DescriptorPool& dpool,
                                 pipeline_handler::DescriptorSet* key_hist_dset,
                                 uint32_t n_elements,
                                 uint32_t n_bit_offset,
                                 gpu_resources::Buffer* src_key_buffer,
                                 gpu_resources::Buffer* src_pos_buffer,
                                 gpu_resources::Buffer* key_hist_buffer,
                                 gpu_resources::Buffer* dst_key_buffer,
                                 gpu_resources::Buffer* dst_pos_buffer)
    : pc_range_(scatter_shader.GeneratePushConstantRanges()[0]),
      n_elements_(n_elements),
      n_bit_offset_(n_bit_offset),
      src_key_buffer_(src_key_buffer),
      src_pos_buffer_(src_pos_buffer),
      key_hist_buffer_(key_hist_buffer),
      dst_key_buffer_(dst_key_buffer),
      dst_pos_buffer_(dst_pos_buffer) {
  pipeline_handler::DescriptorSet* src_buffers_dset =
      scatter_shader.GenerateDescriptorSet(dpool, 0);
  src_buffers_dset->BulkBind(
      std::vector<gpu_resources::Buffer*>{src_key_buffer_, src_pos_buffer_});
  pipeline_handler::DescriptorSet* dst_buffers_dset =
      scatter_shader.GenerateDescriptorSet(dpool, 0);
  src_buffers_dset->BulkBind(
      std::vector<gpu_resources::Buffer*>{dst_key_buffer_, dst_pos_buffer_});
  pipeline_ = pipeline_handler::Compute(
      scatter_shader, {src_buffers_dset, key_hist_dset, dst_buffers_dset});

  gpu_resources::BufferProperties properties{
      .size = sizeof(uint32_t) * n_elements,
      .usage_flags = vk::BufferUsageFlagBits::eStorageBuffer};
  dst_key_buffer_->RequireProperties(properties);
  dst_pos_buffer_->RequireProperties(properties);
}

void SortScatterPass::OnPreRecord() {
  gpu_resources::ResourceAccess access{};
  access.access_flags = vk::AccessFlagBits2KHR::eShaderStorageRead;
  access.stage_flags = vk::PipelineStageFlagBits2KHR::eComputeShader;
  src_key_buffer_->DeclareAccess(access, GetPassIdx());
  src_pos_buffer_->DeclareAccess(access, GetPassIdx());
  key_hist_buffer_->DeclareAccess(access, GetPassIdx());

  access.access_flags |= vk::AccessFlagBits2KHR::eShaderStorageWrite;
  dst_key_buffer_->DeclareAccess(access, GetPassIdx());
  dst_pos_buffer_->DeclareAccess(access, GetPassIdx());
}
void SortScatterPass::OnRecord(vk::CommandBuffer primary_cmd,
                               const std::vector<vk::CommandBuffer>&) {
  RecordSortPhaseInvocation(primary_cmd, pipeline_, pc_range_, n_elements_,
                            n_bit_offset_);
}

void Sort::Apply(render_graph::RenderGraph& render_graph,
                 uint32_t n_elements,
                 gpu_resources::Buffer* key,
                 gpu_resources::Buffer* pos) {
  gpu_resources::Buffer* key_hist =
      render_graph.GetResourceManager().AddBuffer({});
  gpu_resources::Buffer* key_tmp =
      render_graph.GetResourceManager().AddBuffer({});
  gpu_resources::Buffer* pos_tmp =
      render_graph.GetResourceManager().AddBuffer({});

  shader::Loader presort_shader("mp_primitives/sort/block_presort.spv");
  shader::Loader scatter_shader("mp_primitives/sort/scatter.spv");

  auto& dpool = render_graph.GetDescriptorPool();
  pipeline_handler::DescriptorSet* hist_dset =
      presort_shader.GenerateDescriptorSet(dpool, 1);

  for (uint32_t phase_idx = 0; phase_idx < kSortNPhases; phase_idx++) {
    const uint32_t bit_offset = kSortBitsPerPhase * phase_idx;
    block_presort_[phase_idx] =
        SortBlockPresortPass(presort_shader, dpool, hist_dset, n_elements,
                             bit_offset, key, pos, key_hist);
    render_graph.AddPass(&block_presort_[phase_idx],
                         vk::PipelineStageFlagBits2KHR::eComputeShader);
    ScanPass::Apply(render_graph, digit_offset_compute_[phase_idx],
                    GroupsPerInvocation(n_elements) * kNHistBuckets, key_hist);
    scatter_[phase_idx] =
        SortScatterPass(scatter_shader, dpool, hist_dset, n_elements,
                        bit_offset, key, pos, key_hist, key_tmp, pos_tmp);
    render_graph.AddPass(&scatter_[phase_idx],
                         vk::PipelineStageFlagBits2KHR::eComputeShader);
    std::swap(key, key_tmp);
    std::swap(pos, pos_tmp);
  }
  static_assert(false, "COPY RESULT BACK TO THE ORIGINAL BUFFER");
}

}  // namespace detail
}  // namespace mp_primitives
