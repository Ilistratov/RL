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
    std::unique_ptr<Task> task;
  };

  std::vector<vk::CommandBuffer> task_primary_cmd_;
  std::vector<vk::CommandBuffer> task_secondary_cmd_;
  std::vector<TaskInfo> tasks_;
  vk::CommandPool cmd_pool_;

  void CreateCommandBuffers();
  void CleanupTaskWaitInd();

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

  Task* GetTaskByInd(uint32_t task_ind) const;
  const std::vector<uint32_t>& GetTasksToWait(uint32_t dst_task_ind) const;

  void AddBarrier(uint32_t src_task_ind, vk::BufferMemoryBarrier2KHR barrier);
  void AddBarrier(uint32_t src_task_ind, vk::ImageMemoryBarrier2KHR barrier);
  void AddExecutionDependency(uint32_t src_task_ind, uint32_t dst_task_ind);

  uint32_t ScheduleTask(std::unique_ptr<Task> task);

  void PreExecuteInit();
  void Execute();

  ~Executer();
};

}  // namespace gpu_executer
