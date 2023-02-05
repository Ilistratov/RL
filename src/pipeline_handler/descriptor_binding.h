#pragma once

#include <stdint.h>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"

namespace pipeline_handler {

class DescriptorBinding {
 protected:
  vk::DescriptorType type_ = {};
  uint32_t binding_;
  vk::ShaderStageFlags access_stage_flags_;

  DescriptorBinding() = default;
  DescriptorBinding(vk::DescriptorType type,
                    uint32_t binding,
                    vk::ShaderStageFlags access_stage_flags);

 public:
  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept;
  vk::DescriptorType GetType() const noexcept;
  uint32_t GetCount() const noexcept;
  uint32_t GetNum() const noexcept;
  bool IsImageBinding() const noexcept;
  virtual vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<vk::DescriptorImageInfo>& image_info) noexcept = 0;
  virtual bool IsWriteUpdateNeeded() const noexcept = 0;
};

// In it's current implementation can't be used with multiple descriptor sets!
class BufferDescriptorBinding : public DescriptorBinding {
  gpu_resources::Buffer* buffer_to_bind_ = nullptr;

 public:
  BufferDescriptorBinding() = default;
  BufferDescriptorBinding(vk::DescriptorType type,
                          uint32_t binding,
                          vk::ShaderStageFlags access_stage_flags);

  void SetBuffer(gpu_resources::Buffer* buffer,
                 bool update_requierments = false) noexcept;
  void UpdateBufferRequierments(gpu_resources::Buffer* buffer) const;

  vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<vk::DescriptorImageInfo>& image_info) noexcept override;
  bool IsWriteUpdateNeeded() const noexcept override;
};

// In it's current implementation can't be used with multiple descriptor sets!
class ImageDescriptorBinding : public DescriptorBinding {
  gpu_resources::Image* image_to_bind_ = nullptr;

 public:
  ImageDescriptorBinding() = default;
  ImageDescriptorBinding(vk::DescriptorType type,
                         uint32_t binding,
                         vk::ShaderStageFlags access_stage_flags);
  void SetImage(gpu_resources::Image* image,
                bool update_requierments = false) noexcept;
  void UpateImageRequierments(gpu_resources::Image* image) const;
  vk::ImageLayout GetExpectedLayout() const noexcept;

  vk::WriteDescriptorSet GenerateWrite(
      std::vector<vk::DescriptorBufferInfo>& buffer_info,
      std::vector<vk::DescriptorImageInfo>& image_info) noexcept override;
  bool IsWriteUpdateNeeded() const noexcept override;
};

}  // namespace pipeline_handler
