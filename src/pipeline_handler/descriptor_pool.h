#pragma once

#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"

namespace pipeline_handler {

class DescriptorPool {
  std::vector<DescriptorSet> managed_sets_;
  vk::DescriptorPool pool_;
  std::map<vk::DescriptorType, uint32_t> descriptor_type_reserved_count_;

  void CreatePool();
  void AllocateSets();

 public:
  DescriptorPool() = default;

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  DescriptorSet* ReserveDescriptorSet(
      const std::vector<const DescriptorBinding*>& bindings);
  void Create();

  ~DescriptorPool();
};

}  // namespace pipeline_handler
