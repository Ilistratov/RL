#include "render_graph/render_graph.h"

#include "utill/logger.h"

namespace render_graph {

void RenderGraph::AddPass(Pass* pass,
                          vk::Semaphore external_signal,
                          vk::Semaphore external_wait) {
  assert(pass);
  pass->BindResources(passes_.size(), resource_manager_);
  pass->ReserveDescriptorSets(descriptor_pool_);
  executer_.ScheduleTask(pass, pass->GetStageFlags(), external_signal,
                         external_wait, pass->GetSecondaryCmdCount());
  passes_.push_back(std::move(pass));
}

ResourceManager& RenderGraph::GetResourceManager() {
  return resource_manager_;
}

void RenderGraph::Init() {
  LOG(INFO) << "Initializing resources";
  resource_manager_.InitResources();
  LOG(INFO) << "Creating descriptor pool";
  descriptor_pool_.Create();

  LOG(INFO) << "Performing initial layout transitions";
  auto init_barriers_record = [this](vk::CommandBuffer primary_cmd,
                                     const std::vector<vk::CommandBuffer>&) {
    resource_manager_.RecordInitBarriers(primary_cmd);
  };
  gpu_executer::LambdaTask<decltype(init_barriers_record)> init_task(
      init_barriers_record);
  executer_.ExecuteOneTime(&init_task);

  LOG(INFO) << "Notifying passes";
  for (auto& pass : passes_) {
    pass->OnResourcesInitialized();
  }
  LOG(INFO) << "RenderGraph initialized";
}

void RenderGraph::RenderFrame() {
  executer_.Execute();
}

}  // namespace render_graph