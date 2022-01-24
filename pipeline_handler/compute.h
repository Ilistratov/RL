#pragma once

#include <vulkan/vulkan.hpp>

#include "descriptor_handler/binding.h"
#include "descriptor_handler/pool.h"

namespace pipeline_handler {

class Compute {
  vk::Pipeline pipeline_ = {};
  vk::PipelineLayout layout_ = {};
  descriptor_handler::Set* descriptor_set_ = nullptr;

 public:
  Compute() = default;
  Compute(const std::vector<const descriptor_handler::Binding*>& bindings,
          descriptor_handler::Pool& descriptor_pool,
          const std::vector<vk::PushConstantRange>& push_constants,
          const std::string& shader_file_path,
          const std::string& shader_main);

  Compute(const Compute&) = delete;
  void operator=(const Compute&) = delete;

  Compute(Compute&& other) noexcept;
  void operator=(Compute&& other) noexcept;
  void Swap(Compute& other) noexcept;

  void RecordDispatch(vk::CommandBuffer& cmd,
                      uint32_t group_count_x,
                      uint32_t group_count_y,
                      uint32_t group_count_z);

  ~Compute();
};

}  // namespace pipeline_handler
