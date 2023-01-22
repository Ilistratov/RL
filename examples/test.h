#pragma once

#include <stdint.h>
#include <vector>
#include "gpu_resources/buffer.h"
#include "gpu_resources/physical_buffer.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"

namespace examples {

class LoadToGpuPass : public render_graph::Pass {
  gpu_resources::Buffer* staging_ = nullptr;
  gpu_resources::Buffer* values_ = nullptr;
  uint32_t element_count_ = 0;

 public:
  LoadToGpuPass() = default;
  LoadToGpuPass(gpu_resources::Buffer* staging,
                gpu_resources::Buffer* values,
                uint32_t n);

  void OnResourcesInitialized() noexcept override;
  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

struct ScanStageInfo {
  uint32_t stage_spacing = 0;
  uint32_t n_elements = 0;
};

class ScanPass : public render_graph::Pass {
  pipeline_handler::Compute aggregate_pipeline_;
  pipeline_handler::Compute scatter_pipeline_;
  gpu_resources::Buffer* values_ = nullptr;
  pipeline_handler::BufferDescriptorBinding values_aggregate_binding_;
  pipeline_handler::BufferDescriptorBinding values_scatter_binding_;
  ScanStageInfo stage_info_;
  const static vk::PushConstantRange kStageInfoPC;

 public:
  ScanPass() = default;
  ScanPass(gpu_resources::Buffer* values, uint32_t n);
  void OnReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept override;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) noexcept override;
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
  ScanPass scan_;
  LoadToCpuPass load_to_cpu_;

 public:
  TestRenderer();
};

}  // namespace examples
