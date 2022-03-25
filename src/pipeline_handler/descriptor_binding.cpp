#include "pipeline_handler/descriptor_binding.h"

#include "utill/error_handling.h"

namespace pipeline_handler {

vk::WriteDescriptorSet Write::ConvertToVkWrite(
    vk::DescriptorSet dst_set,
    uint32_t binding_id) const& noexcept {
  DCHECK(image_info.empty() || buffer_info.empty())
      << "One and only one resource_info must be presented";
  DCHECK(!image_info.empty() || !buffer_info.empty())
      << "One and only one resource_info must be presented";

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

}  // namespace pipeline_handler
