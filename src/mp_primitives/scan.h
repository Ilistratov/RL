#pragma once

#include <cstdint>

#include "gpu_resources/buffer.h"
#include "pipeline_handler/compute.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"

namespace mp_primitives {

struct ScanStageInfo {
  uint32_t stage_spacing = 0;
  uint32_t n_elements = 0;
  uint32_t is_segmented = 0;
};

class ScanPass : public render_graph::Pass {
  pipeline_handler::Compute aggregate_pipeline_;
  pipeline_handler::Compute scatter_pipeline_;
  gpu_resources::Buffer* values_ = nullptr;
  gpu_resources::Buffer* head_flags_ = nullptr;
  ScanStageInfo stage_info_;
  vk::PushConstantRange stage_info_pc_;

 public:
  ScanPass() = default;
  ScanPass(const shader::Loader& aggregate_shader,
           const shader::Loader& scatter_shader,
           pipeline_handler::DescriptorSet* d_set,
           uint32_t n_elements,
           gpu_resources::Buffer* values,
           gpu_resources::Buffer* segment_heads = nullptr);

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;

  static void Apply(render_graph::RenderGraph& render_graph,
                    ScanPass& dst_pass,
                    uint32_t n_elements,
                    gpu_resources::Buffer* values,
                    gpu_resources::Buffer* head_falgs = nullptr);
};

}  // namespace mp_primitives
