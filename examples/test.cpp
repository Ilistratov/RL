#include "test.h"

#include <concurrencysal.h>
#include <corecrt_memcpy_s.h>
#include <stdint.h>
#include <time.h>
#include <vcruntime_string.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "gpu_resources/buffer.h"
#include "pipeline_handler/compute.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"
#include "shader/loader.h"
#include "utill/logger.h"

namespace examples {

std::vector<int> g_values;

LoadToGpuPass::LoadToGpuPass(gpu_resources::Buffer* staging,
                             gpu_resources::Buffer* values,
                             uint32_t n)
    : staging_(staging), values_(values), element_count_(n) {
  gpu_resources::BufferProperties required_transfer_src_properties{};
  required_transfer_src_properties.memory_flags =
      vk::MemoryPropertyFlagBits::eHostVisible;
  required_transfer_src_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc;
  required_transfer_src_properties.size = n * sizeof(int);
  staging_->RequireProperties(required_transfer_src_properties);
  gpu_resources::BufferProperties required_transfer_dst_properties{};
  required_transfer_dst_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferDst;
  required_transfer_dst_properties.size = n * sizeof(int);
  values_->RequireProperties(required_transfer_dst_properties);
}

void LoadToGpuPass::OnResourcesInitialized() noexcept {
  staging_->LoadDataFromPtr(g_values.data(), element_count_ * sizeof(int), 0);
  auto device = base::Base::Get().GetContext().GetDevice();
  device.flushMappedMemoryRanges(staging_->GetBuffer()->GetMappedMemoryRange());
}

void LoadToGpuPass::OnPreRecord() {
  if (element_count_ == 0) {
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
  gpu_resources::Buffer::RecordCopy(primary_cmd, *staging_, *values_, 0, 0,
                                    element_count_ * sizeof(int));
}

ScanPass::ScanPass(const shader::Loader& aggregate_shader,
                   const shader::Loader& scatter_loader,
                   pipeline_handler::DescriptorSet* d_set,
                   gpu_resources::Buffer* values,
                   uint32_t n)
    : values_(values) {
  stage_info_.stage_spacing = 1;
  stage_info_.n_elements = n;
  gpu_resources::BufferProperties requeired_buffer_propertires{};
  requeired_buffer_propertires.memory_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;
  values_->RequireProperties(requeired_buffer_propertires);
  d_set->GetBufferBinding(0)->UpdateBufferRequierments(values_);
  d_set->GetBufferBinding(0)->SetBuffer(values_);
  aggregate_pipeline_ = pipeline_handler::Compute(aggregate_shader, {d_set});
  scatter_pipeline_ = pipeline_handler::Compute(scatter_loader, {d_set});
  stage_info_pc_ = aggregate_shader.GeneratePushConstantRanges()[0];
}

void ScanPass::OnPreRecord() {
  gpu_resources::ResourceAccess scene_resource_access{};
  scene_resource_access.access_flags =
      vk::AccessFlagBits2KHR::eShaderStorageRead |
      vk::AccessFlagBits2KHR::eShaderStorageWrite;
  scene_resource_access.stage_flags =
      vk::PipelineStageFlagBits2KHR::eComputeShader;
  values_->DeclareAccess(scene_resource_access, GetPassIdx());
}
void ScanPass::OnRecord(vk::CommandBuffer primary_cmd,
                        const std::vector<vk::CommandBuffer>&) noexcept {
  const uint32_t elements_per_thread = 8;
  const uint32_t threads_per_group = 32;
  const uint32_t elements_per_group = elements_per_thread * threads_per_group;
  const vk::BufferMemoryBarrier2KHR buffer_barrier =
      values_->GetBuffer()->GenerateBarrier(
          vk::PipelineStageFlagBits2KHR::eComputeShader,
          vk::AccessFlagBits2KHR::eShaderStorageWrite |
              vk::AccessFlagBits2KHR::eShaderStorageRead,
          vk::PipelineStageFlagBits2KHR::eComputeShader,
          vk::AccessFlagBits2KHR::eShaderStorageWrite |
              vk::AccessFlagBits2KHR::eShaderStorageRead);

  uint32_t head_count =
      (stage_info_.n_elements + elements_per_group - 1) / elements_per_group;
  while (head_count > 0) {
    primary_cmd.pushConstants(aggregate_pipeline_.GetLayout(),
                              stage_info_pc_.stageFlags, stage_info_pc_.offset,
                              stage_info_pc_.size, &stage_info_);
    aggregate_pipeline_.RecordDispatch(primary_cmd, head_count, 1, 1);
    primary_cmd.pipelineBarrier2KHR(
        vk::DependencyInfoKHR({}, {}, buffer_barrier));
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
      primary_cmd.pipelineBarrier2KHR(
          vk::DependencyInfoKHR({}, {}, buffer_barrier));
    }
  }
}

LoadToCpuPass::LoadToCpuPass(gpu_resources::Buffer* staging,
                             gpu_resources::Buffer* values)
    : staging_(staging), values_(values) {
  gpu_resources::BufferProperties required_transfer_src_properties{};
  required_transfer_src_properties.memory_flags =
      vk::MemoryPropertyFlagBits::eHostVisible;
  required_transfer_src_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc;
  values_->RequireProperties(required_transfer_src_properties);
  required_transfer_src_properties.size = g_values.size() * sizeof(int);
  gpu_resources::BufferProperties required_transfer_dst_properties{};
  required_transfer_dst_properties.usage_flags =
      vk::BufferUsageFlagBits::eTransferDst;
  required_transfer_dst_properties.size = g_values.size() * sizeof(int);
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
                                    g_values.size() * sizeof(int));
}

TestRenderer::TestRenderer() {
  gpu_resources::Buffer* staging =
      render_graph_.GetResourceManager().AddBuffer({});
  gpu_resources::Buffer* values =
      render_graph_.GetResourceManager().AddBuffer({});
  g_values = std::vector<int>(256 * 256 * 256 * 2, 1);
  load_to_gpu_ = LoadToGpuPass(staging, values, g_values.size());

  shader::Loader aggregate_shader("shaders/mp-primitives/scan/aggregate.spv");
  shader::Loader scatter_shader("shaders/mp-primitives/scan/scatter.spv");

  pipeline_handler::DescriptorSet* d_set =
      aggregate_shader.GenerateDescriptorSet(render_graph_.GetDescriptorPool(),
                                             0);

  scan_ = ScanPass(aggregate_shader, scatter_shader, d_set, values,
                   g_values.size());
  load_to_cpu_ = LoadToCpuPass(staging, values);
  render_graph_.AddPass(&load_to_gpu_,
                        vk::PipelineStageFlagBits2KHR::eTransfer);
  render_graph_.AddPass(&scan_, vk::PipelineStageFlagBits2KHR::eComputeShader);
  render_graph_.AddPass(&load_to_cpu_,
                        vk::PipelineStageFlagBits2KHR::eTransfer);
  render_graph_.Init();
  LOG << "computation start";
  render_graph_.RenderFrame();
  base::Base::Get().GetContext().GetDevice().waitIdle();
  LOG << "computation end";
  base::Base::Get().GetContext().GetDevice().invalidateMappedMemoryRanges(
      staging->GetBuffer()->GetMappedMemoryRange());
  LOG << "cache invalidated";
  memcpy_s(g_values.data(), g_values.size() * sizeof(int),
           staging->GetBuffer()->GetMappingStart(),
           g_values.size() * sizeof(int));
  LOG << "memcpy done";
  std::ofstream f("a.out");
  for (uint32_t i = 1; i < g_values.size(); i += 1) {
    if (g_values[i] - g_values[i - 1] != 1) {
      f << "Error at index " << i << " expected " << g_values[i - 1] + 1
        << " got " << g_values[i] << '\n';
    }
  }
  LOG << "check done";
  f.close();
}

}  // namespace examples
