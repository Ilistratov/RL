#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"

namespace pipeline_handler {

class DescriptorBinding {
 protected:
  vk::DescriptorType type_ = {};
  vk::ShaderStageFlags descriptor_access_stage_flags_;
  uint32_t dst_array_element_ = {};

  DescriptorBinding(vk::DescriptorType type,
                    vk::ShaderStageFlags descriptor_access_stage_flags,
                    uint32_t dst_array_element);

 public:
  virtual vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept;
  virtual vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<uint32_t>& buffer_info_offset,
      std::vector<vk::DescriptorImageInfo>& image_info,
      std::vector<uint32_t>& image_info_offset) noexcept;
  virtual bool IsWriteUpdateNeeded() const noexcept;
};

class BufferDescriptorBinding : public DescriptorBinding {
  gpu_resources::Buffer* buffer_to_bind_;

  BufferDescriptorBinding(gpu_resources::Buffer* buffer_to_bind,
                          vk::DescriptorType type,
                          vk::ShaderStageFlags descriptor_access_stage_flags,
                          uint32_t dst_array_element = 0);

 public:
  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<uint32_t>& buffer_info_offset,
      std::vector<vk::DescriptorImageInfo>& image_info,
      std::vector<uint32_t>& image_info_offset) noexcept override;
  bool IsWriteUpdateNeeded() const noexcept override;
};

class ImageDescriptorBinding : public DescriptorBinding {
  vk::ImageLayout expected_layout_;
  gpu_resources::Image* image_to_bind_;

  ImageDescriptorBinding(gpu_resources::Image* image_to_bind,
                         vk::DescriptorType type,
                         vk::ShaderStageFlags descriptor_access_stage_flags,
                         vk::ImageLayout expected_layout,
                         uint32_t dst_array_element = 0);

 public:
  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept override;
  vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<uint32_t>& buffer_info_offset,
      std::vector<vk::DescriptorImageInfo>& image_info,
      std::vector<uint32_t>& image_info_offset) noexcept override;
  bool IsWriteUpdateNeeded() const noexcept override;
};

}  // namespace pipeline_handler
