#pragma once

#include <vulkan/vulkan.hpp>

#include "pipeline_handler/binding.h"
#include "pipeline_handler/pool.h"

namespace pipeline_handler {

class Compute {
  vk::Pipeline pipeline_ = {};
  vk::PipelineLayout layout_ = {};
  Set* descriptor_set_ = nullptr;

 public:
  Compute() = default;
  Compute(const std::vector<const Binding*>& bindings,
          Pool& descriptor_pool,
          const std::vector<vk::PushConstantRange>& push_constants,
          const std::string& shader_file_path,
          const std::string& shader_main);

  Compute(const Compute&) = delete;
  void operator=(const Compute&) = delete;

  Compute(Compute&& other) noexcept;
  void operator=(Compute&& other) noexcept;
  void Swap(Compute& other) noexcept;

  void UpdateDescriptorSet(const std::vector<const Binding*>& bindings);
  void RecordDispatch(vk::CommandBuffer& cmd,
                      uint32_t group_count_x,
                      uint32_t group_count_y,
                      uint32_t group_count_z);

  vk::PipelineLayout GetLayout() const;

  ~Compute();
};

}  // namespace pipeline_handler
