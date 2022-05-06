#pragma once

#include <map>
#include <string>

#include "gpu_executer/task.h"
#include "gpu_resources/resource_manager.h"
#include "pipeline_handler/descriptor_pool.h"
#include "render_graph/buffer_pass_bind.h"
#include "render_graph/image_pass_bind.h"

namespace render_graph {

class Pass : public gpu_executer::Task {
  uint32_t user_ind_ = -1;
  uint32_t secondary_cmd_count_ = 0;
  vk::PipelineStageFlags2KHR stage_flags_ = {};
  std::map<std::string, BufferPassBind> buffer_binds_;
  std::map<std::string, ImagePassBind> image_binds_;

  void RecordPostPassParriers(vk::CommandBuffer cmd);

 protected:
  virtual void OnRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) noexcept;

  void AddBuffer(const std::string& buffer_name, BufferPassBind bind);
  void AddImage(const std::string& image_name, ImagePassBind bind);

  gpu_resources::Buffer* GetBuffer(const std::string& buffer_name);
  BufferPassBind& GetBufferPassBind(const std::string& buffer_name);
  gpu_resources::Image* GetImage(const std::string& image_name);
  ImagePassBind& GetImagePassBind(const std::string& image_name);

 public:
  Pass(uint32_t secondary_cmd_count = 0,
       vk::PipelineStageFlags2KHR stage_flags = {});

  void BindResources(uint32_t user_ind,
                     gpu_resources::ResourceManager& resource_manager);
  virtual void ReserveDescriptorSets(
      pipeline_handler::DescriptorPool& pool) noexcept;
  virtual void OnResourcesInitialized() noexcept;

  void OnWorkloadRecord(
      vk::CommandBuffer primary_cmd,
      const std::vector<vk::CommandBuffer>& secondary_cmd) override;

  uint32_t GetSecondaryCmdCount() const;
  vk::PipelineStageFlags2KHR GetStageFlags() const;
};

}  // namespace render_graph
