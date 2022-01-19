#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

namespace descriptor_handler {

struct Write {
  uint32_t dst_array_element = {};
  vk::DescriptorType type = {};
  std::vector<vk::DescriptorImageInfo> image_info = {};
  std::vector<vk::DescriptorBufferInfo> buffer_info = {};

  vk::WriteDescriptorSet ConvertToVkWrite(vk::DescriptorSet dst_set,
                                          uint32_t binding_id) const& noexcept;
};

class Binding {
 public:
  virtual vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept = 0;
  virtual Write GetWrite() const noexcept = 0;
};

}  // namespace descriptor_handler
