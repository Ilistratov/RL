#pragma once

#include <array>
#include <vector>

#include "gpu_resources/buffer.h"
#include "mp_primitives/copy.h"
#include "mp_primitives/scan.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_pool.h"
#include "pipeline_handler/descriptor_set.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"
#include "shader/loader.h"

namespace mp_primitives {
namespace detail {

class SortBlockPresortPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;
  vk::PushConstantRange pc_range_;
  uint32_t n_elements_ = 0;
  uint32_t n_bit_offset_ = 0;
  gpu_resources::Buffer* key_buffer_ = nullptr;
  gpu_resources::Buffer* pos_buffer_ = nullptr;
  gpu_resources::Buffer* key_hist_buffer_ = nullptr;

 public:
  SortBlockPresortPass() = default;
  SortBlockPresortPass(const shader::Loader& presort_shader,
                       pipeline_handler::DescriptorPool& dpool,
                       pipeline_handler::DescriptorSet* key_hist_dset,
                       uint32_t n_elements,
                       uint32_t n_bit_offset,
                       gpu_resources::Buffer* key_buffer,
                       gpu_resources::Buffer* pos_buffer,
                       gpu_resources::Buffer* key_hist_buffer);

  SortBlockPresortPass(SortBlockPresortPass const&) = delete;
  void operator=(SortBlockPresortPass const&) = delete;

  SortBlockPresortPass(SortBlockPresortPass&& other) noexcept;
  void operator=(SortBlockPresortPass&& other) noexcept;
  void Swap(SortBlockPresortPass& other) noexcept;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

class SortScatterPass : public render_graph::Pass {
  pipeline_handler::Compute pipeline_;
  vk::PushConstantRange pc_range_;
  uint32_t n_elements_ = 0;
  uint32_t n_bit_offset_ = 0;
  gpu_resources::Buffer* src_key_buffer_ = nullptr;
  gpu_resources::Buffer* src_pos_buffer_ = nullptr;
  gpu_resources::Buffer* key_hist_buffer_ = nullptr;
  gpu_resources::Buffer* dst_key_buffer_ = nullptr;
  gpu_resources::Buffer* dst_pos_buffer_ = nullptr;

 public:
  SortScatterPass() = default;
  SortScatterPass(const shader::Loader& scatter_shader,
                  pipeline_handler::DescriptorPool& dpool,
                  pipeline_handler::DescriptorSet* key_hist_dset,
                  uint32_t n_elements,
                  uint32_t n_bit_offset,
                  gpu_resources::Buffer* src_key_buffer,
                  gpu_resources::Buffer* src_pos_buffer,
                  gpu_resources::Buffer* key_hist_buffer,
                  gpu_resources::Buffer* dst_key_buffer,
                  gpu_resources::Buffer* dst_pos_buffer);

  SortScatterPass(SortScatterPass const&) = delete;
  void operator=(SortScatterPass const&) = delete;

  SortScatterPass(SortScatterPass&& other) noexcept;
  void operator=(SortScatterPass&& other) noexcept;
  void Swap(SortScatterPass& other) noexcept;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

const uint32_t kSortKeySize = 32;
const uint32_t kSortBitsPerPhase = 4;
const uint32_t kSortNPhases = kSortKeySize / kSortBitsPerPhase;

class Sort {
  std::array<SortBlockPresortPass, kSortNPhases> block_presort_;
  std::array<Scan, kSortNPhases> digit_offset_compute_;
  std::array<SortScatterPass, kSortNPhases> scatter_;
  TinyCopy key_final_copy_;
  TinyCopy pos_final_copy_;

 public:
  Sort() = default;

  Sort(Sort const&) = delete;
  void operator=(Sort const&) = delete;

  Sort(Sort&& other) noexcept;
  void operator=(Sort&& other) noexcept;
  void Swap(Sort& other) noexcept;

  void Apply(render_graph::RenderGraph& render_graph,
             uint32_t n_elements,
             gpu_resources::Buffer* key,
             gpu_resources::Buffer* pos);
};

}  // namespace detail

using detail::Sort;

}  // namespace mp_primitives
