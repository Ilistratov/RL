#include "pipeline_handler/descriptor_binding.h"

#include <vulkan/vulkan_structs.hpp>
#include "utill/error_handling.h"

namespace pipeline_handler {

DescriptorBinding::DescriptorBinding(
    vk::DescriptorType type,
    vk::ShaderStageFlags descriptor_access_stage_flags,
    uint32_t dst_array_element)
    : type_(type),
      descriptor_access_stage_flags_(descriptor_access_stage_flags),
      dst_array_element_(dst_array_element) {}

BufferDescriptorBinding::BufferDescriptorBinding(
    gpu_resources::Buffer* buffer_to_bind,
    vk::DescriptorType type,
    vk::ShaderStageFlags descriptor_access_stage_flags,
    uint32_t dst_array_element)
    : DescriptorBinding(type, descriptor_access_stage_flags, dst_array_element),
      buffer_to_bind_(buffer_to_bind) {
  DCHECK(buffer_to_bind != nullptr) << "Can't bind null buffer";
}

vk::DescriptorSetLayoutBinding BufferDescriptorBinding::GetVkBinding()
    const noexcept {
  return vk::DescriptorSetLayoutBinding(0, type_, 1,
                                        descriptor_access_stage_flags_, {});
}

vk::WriteDescriptorSet BufferDescriptorBinding::GenerateWrite(
    std::vector<vk::DescriptorBufferInfo>& buffer_info,
    std::vector<uint32_t>& buffer_info_offset,
    std::vector<vk::DescriptorImageInfo>&,
    std::vector<uint32_t>& image_info_offset) noexcept {
  DCHECK(buffer_to_bind_ != nullptr) << "Can't bind null buffer";
  DCHECK(!buffer_info_offset.empty())
      << "Offset vector should at least contain 0 offset of 1st element";
  DCHECK(!image_info_offset.empty())
      << "Offset vector should at least contain 0 offset of 1st element";
  buffer_info_offset.push_back(buffer_info_offset.back() + 1);
  buffer_info.push_back(vk::DescriptorBufferInfo{
      buffer_to_bind_->GetBuffer(), 0, buffer_to_bind_->GetSize()});
  image_info_offset.push_back(image_info_offset.back());
  return vk::WriteDescriptorSet{{}, {}, dst_array_element_, 1, type_};
}

bool BufferDescriptorBinding::IsWriteUpdateNeeded() const noexcept {
  return buffer_to_bind_ != nullptr;
}

ImageDescriptorBinding::ImageDescriptorBinding(
    gpu_resources::Image* image_to_bind,
    vk::DescriptorType type,
    vk::ShaderStageFlags descriptor_access_stage_flags,
    vk::ImageLayout expected_layout,
    uint32_t dst_array_element)
    : DescriptorBinding(type, descriptor_access_stage_flags, dst_array_element),
      expected_layout_(expected_layout),
      image_to_bind_(image_to_bind) {
  DCHECK(image_to_bind != nullptr) << "Can't bind null image";
}

vk::DescriptorSetLayoutBinding ImageDescriptorBinding::GetVkBinding()
    const noexcept {
  return vk::DescriptorSetLayoutBinding(0, type_, 1,
                                        descriptor_access_stage_flags_, {});
}

vk::WriteDescriptorSet ImageDescriptorBinding::GenerateWrite(
    std::vector<vk::DescriptorBufferInfo>&,
    std::vector<uint32_t>& buffer_info_offset,
    std::vector<vk::DescriptorImageInfo>& image_info,
    std::vector<uint32_t>& image_info_offset) noexcept {
  DCHECK(image_to_bind_ != nullptr) << "Can't bind null image";
  DCHECK(!buffer_info_offset.empty())
      << "Offset vector should at least contain 0 offset of 1st element";
  DCHECK(!image_info_offset.empty())
      << "Offset vector should at least contain 0 offset of 1st element";
  buffer_info_offset.push_back(buffer_info_offset.back());
  if (!image_to_bind_->GetImageView()) {
    image_to_bind_->CreateImageView();
  }
  image_info_offset.push_back(image_info_offset.back() + 1);
  image_info.push_back(vk::DescriptorImageInfo{
      {}, image_to_bind_->GetImageView(), expected_layout_});
  return vk::WriteDescriptorSet{{}, {}, dst_array_element_, 1, type_};
}

bool ImageDescriptorBinding::IsWriteUpdateNeeded() const noexcept {
  return image_to_bind_ != nullptr;
}

}  // namespace pipeline_handler
