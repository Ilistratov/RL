#pragma once

#include <cstdint>
#include <vector>

#include "gpu_resources/buffer.h"
#include "pipeline_handler/compute.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"

namespace mp_primitives {
namespace detail {

struct ScanStageInfo {
  uint32_t stage_spacing = 0;
  uint32_t n_elements = 0;
  uint32_t is_segmented = 0;
};

class ScanStagePass : public render_graph::Pass {
  vk::PushConstantRange stage_info_pc_;
  pipeline_handler::Compute* pipeline_ = nullptr;
  gpu_resources::Buffer* values_ = nullptr;
  gpu_resources::Buffer* head_flags_ = nullptr;
  uint32_t stage_spacing_ = 0;
  uint32_t n_elements_ = 0;
  uint32_t n_groups_ = 0;

 public:
  ScanStagePass() = default;
  ScanStagePass(vk::PushConstantRange stage_info_pc,
                pipeline_handler::Compute* pipeline,
                uint32_t stage_spacing,
                uint32_t n_elements,
                uint32_t n_groups,
                gpu_resources::Buffer* values,
                gpu_resources::Buffer* head_flags = nullptr);

  ScanStagePass(ScanStagePass const&) = delete;
  void operator=(ScanStagePass const&) = delete;

  ScanStagePass(ScanStagePass&& other) noexcept;
  void operator=(ScanStagePass&& other) noexcept;
  void Swap(ScanStagePass& other) noexcept;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

class Scan {
  pipeline_handler::Compute aggregate_pipeline_;
  pipeline_handler::Compute scatter_pipeline_;
  std::vector<ScanStagePass> passes_;

 public:
  Scan() = default;

  Scan(Scan const&) = delete;
  void operator=(Scan const&) = delete;

  void Apply(render_graph::RenderGraph& render_graph,
             uint32_t n_elements,
             gpu_resources::Buffer* values,
             gpu_resources::Buffer* head_falgs = nullptr);
};

}  // namespace detail

using detail::Scan;

}  // namespace mp_primitives
