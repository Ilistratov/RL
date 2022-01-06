#include "gpu_executer/executer.h"

#include <algorithm>

#include "base/base.h"
#include "utill/logger.h"

namespace gpu_executer {

void Executer::CreateCommandBuffers() {
  LOG(INFO) << "Creating gpu Executer command buffers";
  assert(task_primary_cmd_.empty());
  auto& context = base::Base::Get().GetContext();
  cmd_pool_ = context.GetDevice().createCommandPool(
      vk::CommandPoolCreateInfo({}, context.GetQueueFamilyIndex()));
  task_primary_cmd_ =
      context.GetDevice().allocateCommandBuffers(vk::CommandBufferAllocateInfo(
          cmd_pool_, vk::CommandBufferLevel::ePrimary, tasks_.size()));
  task_secondary_cmd_ =
      context.GetDevice().allocateCommandBuffers(vk::CommandBufferAllocateInfo(
          cmd_pool_, vk::CommandBufferLevel::eSecondary, 3 * tasks_.size()));

  for (uint32_t task_id = 0; task_id < tasks_.size(); task_id++) {
    std::array<vk::CommandBuffer, 3> task_secondary_cmd = {
        task_secondary_cmd_[task_id * 3 + 0],
        task_secondary_cmd_[task_id * 3 + 1],
        task_secondary_cmd_[task_id * 3 + 2]};
    tasks_[task_id].task->OnCmdCreate(task_primary_cmd_[task_id],
                                      task_secondary_cmd);
  }
}

void Executer::NotifyRecord() {
  LOG(INFO) << "Notifying gpu Executer tasks";
  for (auto& task : tasks_) {
    task.task->OnRecord();
  }
}

void Executer::OrderTasksDfs(uint32_t task_ind,
                             std::vector<uint32_t>& used_flag) {
  used_flag[task_ind] = 1;
  for (auto dep_task_ind : tasks_[task_ind].dep_graph_) {
    if (used_flag[dep_task_ind] == 1) {
      LOG(ERROR) << "Dependency loop for tasks " << task_ind << " "
                 << dep_task_ind;
      assert(false);
    }
    if (!used_flag[dep_task_ind]) {
      OrderTasksDfs(dep_task_ind, used_flag);
    }
  }
  used_flag[task_ind] = 2;
  task_order_.push_back(task_ind);
}

void Executer::OrderTasks() {
  LOG(INFO) << "Ordering gpu Executer tasks";
  assert(task_order_.empty());
  task_order_.reserve(tasks_.size());
  std::vector<uint32_t> used_flag(tasks_.size());
  for (uint32_t task_ind = 0; task_ind < tasks_.size(); task_ind++) {
    OrderTasksDfs(task_ind, used_flag);
  }
  std::reverse(task_order_.begin(), task_order_.end());
  LOG(DEBUG) << "Ordered gpu Executer tasks, order: " << task_order_;
}

void Executer::CleanupDepGraph() {
  LOG(INFO) << "Cleaning up gpu Executer dep graph";
  for (auto& task : tasks_) {
    task.dep_graph_.clear();
    task.dep_graph_.shrink_to_fit();
  }
  dep_graph_edges_.clear();
}

void Executer::CleanupTaskWaitInd() {
  LOG(INFO) << "Cleaning up gpu Executer tasks wait ind";
  for (auto& task : tasks_) {
    std::sort(task.wait_task_ind.begin(), task.wait_task_ind.end());
    auto it = std::unique(task.wait_task_ind.begin(), task.wait_task_ind.end());
    task.wait_task_ind.resize(it - task.wait_task_ind.begin());
    task.wait_task_ind.shrink_to_fit();
  }
}

void Executer::AddDepGraphEdge(uint32_t src_task_ind, uint32_t dst_task_ind) {
  assert(src_task_ind < tasks_.size());
  assert(dst_task_ind < tasks_.size());
  if (dep_graph_edges_.contains({src_task_ind, dst_task_ind})) {
    return;
  }
  dep_graph_edges_.insert({src_task_ind, dst_task_ind});
  tasks_[dst_task_ind].dep_graph_.push_back(src_task_ind);
}

Executer::SubmitInfo Executer::GetBatchSubmitInfo(uint32_t batch_start,
                                                  uint32_t batch_end) {
  SubmitInfo result;
  for (uint32_t task_pos = batch_start; task_pos < batch_end; ++task_pos) {
    uint32_t task_ind = task_order_[task_pos];
    auto& task = tasks_[task_ind];
    result.cmd_to_execute.push_back(task_primary_cmd_[task_ind]);

    auto semaphores_to_wait = task.task->GetSemaphoresToWait();
    assert(semaphores_to_wait.empty() || result.semaphores_to_wait.empty());
    result.semaphores_to_wait = semaphores_to_wait;

    auto semaphores_to_signal = task.task->GetSemaphoresToSignal();
    for (auto semaphore_submit_info : semaphores_to_signal) {
      result.semaphores_to_signal.push_back(semaphore_submit_info);
    }
  }
  return result;
}

std::vector<vk::BufferMemoryBarrier2KHR> Executer::GetBufferBarriers(
    const std::vector<uint32_t>& barrier_ind) const {
  std::vector<vk::BufferMemoryBarrier2KHR> result;
  result.reserve(barrier_ind.size());
  for (uint32_t ind : barrier_ind) {
    result.push_back(buffer_barriers_[ind]);
  }
  return result;
}

std::vector<vk::ImageMemoryBarrier2KHR> Executer::GetImageBarriers(
    const std::vector<uint32_t>& barrier_ind) const {
  std::vector<vk::ImageMemoryBarrier2KHR> result;
  result.reserve(barrier_ind.size());
  for (uint32_t ind : barrier_ind) {
    result.push_back(image_barriers_[ind]);
  }
  return result;
}

Task* Executer::GetTaskByInd(uint32_t id) const {
  return tasks_.at(id).task.get();
}

const std::vector<uint32_t>& Executer::GetTasksToWait(
    uint32_t dst_task_ind) const {
  assert(dst_task_ind < tasks_.size());
  return tasks_[dst_task_ind].wait_task_ind;
}

void Executer::AddBarrierDependency(uint32_t src_task_ind,
                                    vk::BufferMemoryBarrier2KHR barrier,
                                    uint32_t dst_task_ind) {
  assert(src_task_ind < tasks_.size());
  assert(dst_task_ind < tasks_.size());
  uint32_t barrier_ind = buffer_barriers_.size();
  buffer_barriers_.push_back(barrier);
  tasks_[src_task_ind]
      .task->GetSrcDep(dst_task_ind)
      .buffer_barrier_ind.push_back(barrier_ind);
  tasks_[dst_task_ind]
      .task->GetDstDep(src_task_ind)
      .buffer_barrier_ind.push_back(barrier_ind);
  AddDepGraphEdge(src_task_ind, dst_task_ind);
}

void Executer::AddBarrierDependency(uint32_t src_task_ind,
                                    vk::ImageMemoryBarrier2KHR barrier,
                                    uint32_t dst_task_ind) {
  assert(src_task_ind < tasks_.size());
  assert(dst_task_ind < tasks_.size());
  uint32_t barrier_ind = image_barriers_.size();
  image_barriers_.push_back(barrier);
  tasks_[src_task_ind]
      .task->GetSrcDep(dst_task_ind)
      .image_barrier_ind.push_back(barrier_ind);
  tasks_[dst_task_ind]
      .task->GetDstDep(src_task_ind)
      .image_barrier_ind.push_back(barrier_ind);
  AddDepGraphEdge(src_task_ind, dst_task_ind);
}

void Executer::AddExecutionDependency(uint32_t src_task_ind,
                                      uint32_t dst_task_ind) {
  tasks_[dst_task_ind].wait_task_ind.push_back(src_task_ind);
  AddDepGraphEdge(src_task_ind, dst_task_ind);
}

uint32_t Executer::ScheduleTask(std::unique_ptr<Task> task) {
  uint32_t task_ind = tasks_.size();
  tasks_.push_back({});
  auto& task_info = tasks_[task_ind];
  task_info.task = std::move(task);
  task_info.task->OnSchedule(this, task_ind);
  return task_ind;
}

void Executer::PreExecuteInit() {
  LOG(INFO) << "Initializing gpu Executer";
  assert(!tasks_.empty());
  CreateCommandBuffers();
  NotifyRecord();
  OrderTasks();
  CleanupDepGraph();
  CleanupTaskWaitInd();
  LOG(INFO) << "Initialized gpu Executer";
}

void Executer::Execute() {
  std::vector<Executer::SubmitInfo> batch;
  std::vector<vk::SubmitInfo2KHR> batch_submit_info;
  uint32_t batch_start_pos = 0;

  while (batch_start_pos < tasks_.size()) {
    uint32_t batch_end_pos = batch_start_pos;
    while (batch_end_pos + 1 < tasks_.size() &&
           !tasks_[task_order_[batch_end_pos]].task->IsHasExecutionDep()) {
      ++batch_end_pos;
    }
    ++batch_end_pos;

    batch.push_back(GetBatchSubmitInfo(batch_start_pos, batch_end_pos));
    batch_start_pos = batch_end_pos;
    batch_submit_info.push_back(vk::SubmitInfo2KHR(
        {}, batch.back().semaphores_to_wait, batch.back().cmd_to_execute,
        batch.back().semaphores_to_signal));
  }

  auto& context = base::Base::Get().GetContext();
  context.GetQueue(0).submit2KHR(batch_submit_info);
}

Executer::~Executer() {
  for (auto& task : tasks_) {
    task.task->WaitOnCompletion();
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.freeCommandBuffers(cmd_pool_, task_primary_cmd_);
  device.freeCommandBuffers(cmd_pool_, task_secondary_cmd_);
  device.destroyCommandPool(cmd_pool_);
}

}  // namespace gpu_executer
