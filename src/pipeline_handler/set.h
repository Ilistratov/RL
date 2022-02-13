#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "pipeline_handler/binding.h"

namespace pipeline_handler {

class Set {
  vk::DescriptorSetLayout layout_;
  vk::DescriptorSet set_;

  friend class Pool;

 public:
  Set(const std::vector<const Binding*>& bindings);

  Set(const Set&) = delete;
  Set& operator=(const Set&) = delete;

  vk::DescriptorSetLayout GetLayout() const;
  vk::DescriptorSet GetSet() const;

  void UpdateDescriptorSet(const std::vector<const Binding*>& bindings) const;

  ~Set();
};

}  // namespace pipeline_handler
