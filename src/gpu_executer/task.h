#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

namespace gpu_executer {

class Executer;

class Task {
 public:
  virtual void OnWorkloadRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) = 0;
  virtual ~Task() = default;
};

template <typename T>
concept CmdInvocable = std::is_invocable < T,
        vk::CommandBuffer, const std::vector<vk::CommandBuffer>
& > ::value;

template <CmdInvocable Func>
class LambdaTask : public Task {
  Func func_;

 public:
  template <typename F>
  LambdaTask(F&& func) : func_(std::forward<F>(func)) {}

  void OnWorkloadRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) override {
    func_(primary_cmd, secondary_cmd);
  }
};

}  // namespace gpu_executer
