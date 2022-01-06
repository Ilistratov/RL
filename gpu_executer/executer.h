#pragma once

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_executer/task.h"

namespace gpu_executer {

class Executer {
  struct TaskInfo {
    std::vector<uint32_t> wait_task_ind;
    std::vector<uint32_t> dep_graph_;
    std::unique_ptr<Task> task;
  };

  std::vector<vk::BufferMemoryBarrier2KHR> buffer_barriers_;
  std::vector<vk::ImageMemoryBarrier2KHR> image_barriers_;
  std::vector<vk::CommandBuffer> task_primary_cmd_;
  std::vector<vk::CommandBuffer> task_secondary_cmd_;
  std::set<std::pair<uint32_t, uint32_t>> dep_graph_edges_;
  std::vector<uint32_t> task_order_;
  std::vector<TaskInfo> tasks_;
  vk::CommandPool cmd_pool_;

  void CreateCommandBuffers();
  void CreateEvents();
  void NotifyRecord();
  void OrderTasksDfs(uint32_t task_ind, std::vector<uint32_t>& used_flag);
  void OrderTasks();
  void CleanupDepGraph();
  void CleanupTaskWaitInd();

  void AddDepGraphEdge(uint32_t src_task_ind, uint32_t dst_task_ind);

  struct SubmitInfo {
    std::vector<vk::SemaphoreSubmitInfoKHR> semaphores_to_wait;
    std::vector<vk::CommandBufferSubmitInfoKHR> cmd_to_execute;
    std::vector<vk::SemaphoreSubmitInfoKHR> semaphores_to_signal;
  };

  SubmitInfo GetBatchSubmitInfo(uint32_t batch_start, uint32_t batch_end);

 public:
  Executer() = default;

  Executer(const Executer&) = delete;
  void operator=(const Executer&) = delete;

  std::vector<vk::BufferMemoryBarrier2KHR> GetBufferBarriers(
      const std::vector<uint32_t>& barrier_ind) const;
  std::vector<vk::ImageMemoryBarrier2KHR> GetImageBarriers(
      const std::vector<uint32_t>& barrier_ind) const;
  Task* GetTaskByInd(uint32_t task_ind) const;
  const std::vector<uint32_t>& GetTasksToWait(uint32_t dst_task_ind) const;

  void AddBarrierDependency(uint32_t src_task_ind,
                            vk::BufferMemoryBarrier2KHR barrier,
                            uint32_t dst_task_ind);
  void AddBarrierDependency(uint32_t src_task_ind,
                            vk::ImageMemoryBarrier2KHR barrier,
                            uint32_t dst_task_ind);
  void AddExecutionDependency(uint32_t src_task_ind, uint32_t dst_task_ind);

  uint32_t ScheduleTask(std::unique_ptr<Task> task);

  void PreExecuteInit();
  void Execute();

  ~Executer();
};

}  // namespace gpu_executer
