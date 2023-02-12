#pragma once

#include <stdint.h>
#include <vector>

#include "gpu_resources/buffer.h"
#include "mp_primitives/scan.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"
#include "shader/loader.h"

namespace examples {

class LoadToGpuPass : public render_graph::Pass {
  gpu_resources::Buffer* staging_ = nullptr;
  gpu_resources::Buffer* values_ = nullptr;
  gpu_resources::Buffer* head_flags_ = nullptr;

 public:
  LoadToGpuPass() = default;
  LoadToGpuPass(gpu_resources::Buffer* staging,
                gpu_resources::Buffer* values,
                gpu_resources::Buffer* head_flags);

  void OnResourcesInitialized() noexcept override;
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

struct ScanStageInfo {
  uint32_t stage_spacing = 0;
  uint32_t n_elements = 0;
};

class LoadToCpuPass : public render_graph::Pass {
  gpu_resources::Buffer* staging_ = nullptr;
  gpu_resources::Buffer* values_ = nullptr;

 public:
  LoadToCpuPass() = default;
  LoadToCpuPass(gpu_resources::Buffer* staging, gpu_resources::Buffer* values);

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

class TestRenderer {
  render_graph::RenderGraph render_graph_;
  LoadToGpuPass load_to_gpu_;
  mp_primitives::ScanPass scan_;
  LoadToCpuPass load_to_cpu_;

 public:
  TestRenderer();
};

}  // namespace examples
