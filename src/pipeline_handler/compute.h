#pragma once

#include <glm/fwd.hpp>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_pool.h"
#include "pipeline_handler/descriptor_set.h"
#include "shader/loader.h"

namespace pipeline_handler {

class Compute {
  vk::Pipeline pipeline_ = {};
  vk::PipelineLayout layout_ = {};
  std::vector<DescriptorSet*> descriptor_sets_;

 public:
  Compute() = default;
  Compute(const shader::Loader& loader,
          std::vector<DescriptorSet*> descriptor_sets,
          const std::string& entry_point = "main");

  Compute(const Compute&) = delete;
  void operator=(const Compute&) = delete;

  Compute(Compute&& other) noexcept;
  void operator=(Compute&& other) noexcept;
  void Swap(Compute& other) noexcept;

  void RecordDispatch(vk::CommandBuffer& cmd,
                      uint32_t group_count_x,
                      uint32_t group_count_y,
                      uint32_t group_count_z);

  vk::PipelineLayout GetLayout() const;

  ~Compute();
};

}  // namespace pipeline_handler
