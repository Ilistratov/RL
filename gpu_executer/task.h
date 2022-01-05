#pragma once

#include <array>
#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_executer/timeline_semaphore.h"

namespace gpu_executer {

class Executer;

class Task {
 public:
  struct BarrierDep {
    std::vector<uint32_t> buffer_barrier_ind;
    std::vector<uint32_t> image_barrier_ind;
  };

 private:
  std::map<uint32_t, BarrierDep> src_deps_by_task_ind_;
  std::map<uint32_t, vk::Event> src_events_by_task_ind_;
  std::map<uint32_t, BarrierDep> dst_deps_by_task_ind_;
  TimelineSemaphore on_complete_semaphore_;
  vk::CommandBuffer src_dep_cmd_;
  vk::CommandBuffer workload_cmd_;
  vk::CommandBuffer dst_dep_cmd_;
  vk::CommandBuffer primary_cmd_;
  vk::PipelineStageFlags2KHR stage_flags_;
  vk::Semaphore external_wait_;
  vk::Semaphore external_signal_;
  Executer* executer_ = nullptr;
  uint32_t task_ind_ = UINT32_MAX;

  void RecordSrcDep(uint32_t dst_task_ind, const BarrierDep& dep) const;
  void RecordDstDep(uint32_t src_task_ind, const BarrierDep& dep) const;

 public:
  Task(vk::PipelineStageFlags2KHR stage_flags,
       vk::Semaphore external_wait = {},
       vk::Semaphore external_signal = {});

  BarrierDep& GetSrcDep(uint32_t dst_task_ind);
  BarrierDep& GetDstDep(uint32_t src_task_ind);

  bool IsHasExecutionDep() const;
  std::vector<vk::SemaphoreSubmitInfoKHR> GetSemaphoresToWait();
  std::vector<vk::SemaphoreSubmitInfoKHR> GetSemaphoresToSignal();
  void WaitOnCompletion();

  void OnSchedule(Executer* executer, uint32_t task_ind);
  void OnCmdCreate(vk::CommandBuffer primary_cmd,
                   std::array<vk::CommandBuffer, 3> secondary_cmd);
  void OnRecord();
  virtual void OnWorkloadRecord(vk::CommandBuffer cmd) = 0;

  virtual ~Task();
};

}  // namespace gpu_executer
