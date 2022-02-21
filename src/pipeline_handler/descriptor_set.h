#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "pipeline_handler/descriptor_binding.h"

namespace pipeline_handler {

class DescriptorSet {
  vk::DescriptorSetLayout layout_ = {};
  vk::DescriptorSet set_ = {};

  friend class DescriptorPool;

 public:
  DescriptorSet() = default;
  DescriptorSet(const std::vector<const DescriptorBinding*>& bindings);

  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&& other) noexcept;
  void operator=(DescriptorSet&& other) noexcept;
  void Swap(DescriptorSet& other) noexcept;

  vk::DescriptorSetLayout GetLayout() const;
  vk::DescriptorSet GetSet() const;

  void UpdateDescriptorSet(
      const std::vector<const DescriptorBinding*>& bindings) const;

  ~DescriptorSet();
};

}  // namespace pipeline_handler
