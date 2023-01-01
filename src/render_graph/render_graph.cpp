#include "render_graph/render_graph.h"

#include <vulkan/vulkan_enums.hpp>
#include "gpu_executer/task.h"
#include "gpu_resources/pass_access_syncronizer.h"
#include "gpu_resources/resource_manager.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace render_graph {

PreFrameResourceInitializerTask::PreFrameResourceInitializerTask(
    gpu_resources::PassAccessSyncronizer* access_syncronizer,
    uint32_t pass_count)
    : access_syncronizer_(access_syncronizer), pass_count_(pass_count) {}

void PreFrameResourceInitializerTask::OnWorkloadRecord(
    vk::CommandBuffer cmd,
    const std::vector<vk::CommandBuffer>&) {
  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers_ =
      access_syncronizer_->GetBufferPostPassBarriers(pass_count_);
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers_ =
      access_syncronizer_->GetImagePostPassBarriers(pass_count_);
  vk::DependencyInfoKHR dep_info({}, {}, buffer_barriers_, image_barriers_);
  cmd.pipelineBarrier2KHR(dep_info);
}

RenderGraph::RenderGraph() {
  executer_.ScheduleTask(&initialize_task_,
                         vk::PipelineStageFlagBits2KHR::eTopOfPipe);
}

void RenderGraph::AddPass(Pass* pass,
                          vk::PipelineStageFlags2KHR stage_flags,
                          vk::Semaphore external_signal,
                          vk::Semaphore external_wait) {
  DCHECK(pass) << "Can't add null";
  pass->OnRegister(passes_.size(), resource_manager_.GetAccessSyncronizer(),
                   descriptor_pool_);
  executer_.ScheduleTask(pass, stage_flags, external_signal, external_wait,
                         pass->GetSecondaryCmdCount());
  passes_.push_back(pass);
}

gpu_resources::ResourceManager& RenderGraph::GetResourceManager() {
  return resource_manager_;
}

void RenderGraph::Init() {
  LOG << "Initializing resources";
  initialize_task_ = PreFrameResourceInitializerTask(
      resource_manager_.GetAccessSyncronizer(), passes_.size());
  resource_manager_.InitResources(passes_.size());
  LOG << "Creating descriptor pool";
  descriptor_pool_.Create();

  LOG << "Notifying passes";
  for (auto& pass : passes_) {
    pass->OnResourcesInitialized();
  }
  LOG << "RenderGraph initialized";
}

void RenderGraph::RenderFrame() {
  for (Pass* pass : passes_) {
    pass->OnPreRecord();
  }
  executer_.Execute();
}

}  // namespace render_graph
