#include "descriptor_handler/binding.h"

#include <cassert>

namespace descriptor_handler {

vk::WriteDescriptorSet Write::ConvertToVkWrite(
    vk::DescriptorSet dst_set,
    uint32_t binding_id) const& noexcept {
  assert(image_info.empty() || buffer_info.empty());
  assert(!image_info.empty() || !buffer_info.empty());

  vk::WriteDescriptorSet result;
  result.dstSet = dst_set;
  result.dstBinding = binding_id;
  result.dstArrayElement = dst_array_element;
  result.descriptorType = type;

  if (!buffer_info.empty()) {
    result.descriptorCount = (uint32_t)buffer_info.size();
    result.pBufferInfo = buffer_info.data();
  } else if (!image_info.empty()) {
    result.descriptorCount = (uint32_t)image_info.size();
    result.pImageInfo = image_info.data();
  }
  return result;
}

}  // namespace descriptor_handler
