#include "render_graph/render_graph.h"

#include "utill/error_handling.h"
#include "utill/logger.h"

namespace render_graph {

void RenderGraph::AddPass(Pass* pass,
                          vk::PipelineStageFlags2KHR stage_flags,
                          vk::Semaphore external_signal,
                          vk::Semaphore external_wait) {
  DCHECK(pass) << "Can't add null";
  pass->BindResources(passes_.size(), resource_manager_);
  pass->ReserveDescriptorSets(descriptor_pool_);
  executer_.ScheduleTask(pass, stage_flags, external_signal, external_wait,
                         pass->GetSecondaryCmdCount());
  passes_.push_back(std::move(pass));
}

gpu_resources::ResourceManager& RenderGraph::GetResourceManager() {
  return resource_manager_;
}

void RenderGraph::Init() {
  LOG << "Initializing resources";
  resource_manager_.InitResources();
  LOG << "Creating descriptor pool";
  descriptor_pool_.Create();

  LOG << "Performing initial layout transitions";
  auto init_barriers_record = [this](vk::CommandBuffer primary_cmd,
                                     const std::vector<vk::CommandBuffer>&) {
    resource_manager_.RecordInitBarriers(primary_cmd);
  };
  gpu_executer::LambdaTask<decltype(init_barriers_record)> init_task(
      init_barriers_record);
  executer_.ExecuteOneTime(&init_task);

  LOG << "Notifying passes";
  for (auto& pass : passes_) {
    pass->OnResourcesInitialized();
  }
  LOG << "RenderGraph initialized";
}

void RenderGraph::RenderFrame() {
  executer_.Execute();
}

}  // namespace render_graph
