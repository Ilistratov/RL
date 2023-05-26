#pragma once

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_executor/command_pool.h"
#include "gpu_executor/task.h"

namespace gpu_executor {

class Executor {
  struct TaskInfo {
    Task* task;
    vk::PipelineStageFlags2KHR stage_flags = {};
    vk::Semaphore external_signal = {};
    vk::Semaphore external_wait = {};
    uint32_t secondary_cmd_count = 0;

    bool HasSemaphoreOperations() const;
  };

  CommandPool cmd_pool_;
  std::vector<TaskInfo> tasks_;

  struct SubmitInfo {
    std::vector<vk::SemaphoreSubmitInfoKHR> semaphore_to_wait;
    vk::CommandBufferSubmitInfoKHR cmd_to_execute;
    std::vector<vk::CommandBuffer> secondary_cmd;
    std::vector<vk::SemaphoreSubmitInfoKHR> semaphore_to_signal;
  };

  SubmitInfo RecordCmdBatch(uint32_t batch_start, uint32_t batch_end);

 public:
  Executor() = default;

  Executor(const Executor&) = delete;
  void operator=(const Executor&) = delete;

  void ScheduleTask(Task* task,
                    vk::PipelineStageFlags2KHR stage_flags = {},
                    vk::Semaphore external_signal = {},
                    vk::Semaphore external_wait = {},
                    uint32_t secondary_cmd_count = 0);

  void ExecuteOneTime(Task* task, uint32_t secondary_cmd_count = 0);

  void Execute();
};

}  // namespace gpu_executor
