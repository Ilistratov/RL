#include "render_graph/render_graph.h"

#include "utill/error_handling.h"
#include "utill/logger.h"

namespace render_graph {

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
