#include "pipeline_handler/descriptor_binding.h"

#include <stdint.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_to_string.hpp>
#include "gpu_resources/physical_buffer.h"
#include "gpu_resources/physical_image.h"
#include "utill/error_handling.h"

namespace pipeline_handler {

namespace {

vk::BufferUsageFlagBits DescriptorTypeToBufferUsage(
    vk::DescriptorType descriptor_type) {
  switch (descriptor_type) {
    case vk::DescriptorType::eUniformTexelBuffer:
      return vk::BufferUsageFlagBits::eUniformTexelBuffer;

    case vk::DescriptorType::eStorageTexelBuffer:
      return vk::BufferUsageFlagBits::eStorageTexelBuffer;

    case vk::DescriptorType::eUniformBufferDynamic:
    case vk::DescriptorType::eUniformBuffer:
      return vk::BufferUsageFlagBits::eUniformBuffer;

    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eStorageBufferDynamic:
      return vk::BufferUsageFlagBits::eStorageBuffer;

    case vk::DescriptorType::eAccelerationStructureKHR:
    case vk::DescriptorType::eAccelerationStructureNV:
      return vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR;

    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eStorageImage:
      return {};

    case vk::DescriptorType::eInputAttachment:
    case vk::DescriptorType::eInlineUniformBlock:
    case vk::DescriptorType::eSampler:
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eBlockMatchImageQCOM:
    case vk::DescriptorType::eSampleWeightImageQCOM:
    case vk::DescriptorType::eMutableEXT:
    default:
      DCHECK(false) << "Unsupported descriptor type: "
                    << vk::to_string(descriptor_type);
      break;
  }
  return {};
}

vk::ImageUsageFlagBits DescriptorTypeToImageUsage(
    vk::DescriptorType descriptor_type) {
  switch (descriptor_type) {
    case vk::DescriptorType::eSampledImage:
      return vk::ImageUsageFlagBits::eSampled;

    case vk::DescriptorType::eStorageImage:
      return vk::ImageUsageFlagBits::eStorage;

    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageTexelBuffer:
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eUniformBufferDynamic:
    case vk::DescriptorType::eStorageBufferDynamic:
    case vk::DescriptorType::eAccelerationStructureKHR:
    case vk::DescriptorType::eAccelerationStructureNV:
      return {};

    case vk::DescriptorType::eInputAttachment:
    case vk::DescriptorType::eInlineUniformBlock:
    case vk::DescriptorType::eMutableEXT:
    case vk::DescriptorType::eSampleWeightImageQCOM:
    case vk::DescriptorType::eBlockMatchImageQCOM:
    case vk::DescriptorType::eSampler:
    case vk::DescriptorType::eCombinedImageSampler:
    default:
      DCHECK(false) << "Unsupported descriptor type: "
                    << vk::to_string(descriptor_type);
      break;
  }
  return {};
}

vk::ImageLayout ImageUsageToLayout(vk::ImageUsageFlagBits image_usage) {
  switch (image_usage) {
    case vk::ImageUsageFlagBits::eTransferSrc:
      return vk::ImageLayout::eTransferSrcOptimal;

    case vk::ImageUsageFlagBits::eTransferDst:
      return vk::ImageLayout::eTransferDstOptimal;

    case vk::ImageUsageFlagBits::eSampled:
      return vk::ImageLayout::eReadOnlyOptimal;

    case vk::ImageUsageFlagBits::eStorage:
      return vk::ImageLayout::eGeneral;

    case vk::ImageUsageFlagBits::eColorAttachment:
      return vk::ImageLayout::eColorAttachmentOptimal;

    case vk::ImageUsageFlagBits::eDepthStencilAttachment:
      return vk::ImageLayout::eDepthAttachmentOptimal;

    case vk::ImageUsageFlagBits::eTransientAttachment:
    case vk::ImageUsageFlagBits::eInputAttachment:
    case vk::ImageUsageFlagBits::eFragmentDensityMapEXT:
    case vk::ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR:
    case vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT:
    case vk::ImageUsageFlagBits::eInvocationMaskHUAWEI:
    case vk::ImageUsageFlagBits::eSampleWeightQCOM:
    case vk::ImageUsageFlagBits::eSampleBlockMatchQCOM:
    default:
      DCHECK(false) << "Unsupported usage type: " << vk::to_string(image_usage);
      break;
  }
  return vk::ImageLayout::eGeneral;
}

}  // namespace

DescriptorBinding::DescriptorBinding(vk::DescriptorType type,
                                     uint32_t binding,
                                     vk::ShaderStageFlags access_stage_flags)
    : type_(type), binding_(binding), access_stage_flags_(access_stage_flags) {}

vk::DescriptorSetLayoutBinding DescriptorBinding::GetVkBinding()
    const noexcept {
  return vk::DescriptorSetLayoutBinding(binding_, type_, 1, access_stage_flags_,
                                        {});
}

vk::DescriptorType DescriptorBinding::GetType() const noexcept {
  return type_;
}

uint32_t DescriptorBinding::GetCount() const noexcept {
  return 1;  // update when descriptors for array are supported
}

uint32_t DescriptorBinding::GetNum() const noexcept {
  return binding_;
}

bool DescriptorBinding::IsImageBinding() const noexcept {
  switch (type_) {
    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eStorageImage:
      return true;

    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageTexelBuffer:
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eUniformBufferDynamic:
    case vk::DescriptorType::eStorageBufferDynamic:
    case vk::DescriptorType::eAccelerationStructureKHR:
    case vk::DescriptorType::eAccelerationStructureNV:
      return false;

    case vk::DescriptorType::eInputAttachment:
    case vk::DescriptorType::eInlineUniformBlock:
    case vk::DescriptorType::eMutableEXT:
    case vk::DescriptorType::eSampleWeightImageQCOM:
    case vk::DescriptorType::eBlockMatchImageQCOM:
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eSampler:
    default:
      DCHECK(false) << "Unsupported descriptor type: " << vk::to_string(type_);
      break;
  }
  return false;
}

BufferDescriptorBinding::BufferDescriptorBinding(
    vk::DescriptorType type,
    uint32_t binding,
    vk::ShaderStageFlags access_stage_flags)
    : DescriptorBinding(type, binding, access_stage_flags) {}

void BufferDescriptorBinding::SetBuffer(gpu_resources::Buffer* buffer,
                                        bool update_requierments) noexcept {
  buffer_to_bind_ = buffer;
  if (update_requierments && buffer) {
    UpdateBufferRequierments(buffer);
  }
}

void BufferDescriptorBinding::UpdateBufferRequierments(
    gpu_resources::Buffer* buffer) const {
  gpu_resources::BufferProperties properties;
  properties.usage_flags |= DescriptorTypeToBufferUsage(type_);
  DCHECK(buffer);
  buffer->RequireProperties(properties);
}

vk::WriteDescriptorSet BufferDescriptorBinding::GenerateWrite(
    std::vector<vk::DescriptorBufferInfo>& buffer_info,
    std::vector<vk::DescriptorImageInfo>&) noexcept {
  DCHECK(buffer_to_bind_ != nullptr) << "Can't bind null buffer";
  buffer_info.push_back(vk::DescriptorBufferInfo{
      buffer_to_bind_->GetVkBuffer(), 0, buffer_to_bind_->GetSize()});
  buffer_to_bind_ = nullptr;
  return vk::WriteDescriptorSet{{}, binding_, 0, 1, type_};
}

bool BufferDescriptorBinding::IsWriteUpdateNeeded() const noexcept {
  return buffer_to_bind_ != nullptr;
}

ImageDescriptorBinding::ImageDescriptorBinding(
    vk::DescriptorType type,
    uint32_t binding,
    vk::ShaderStageFlags access_stage_flags)
    : DescriptorBinding(type, binding, access_stage_flags) {}

void ImageDescriptorBinding::SetImage(gpu_resources::Image* image,
                                      bool update_requierments) noexcept {
  image_to_bind_ = image;
  if (update_requierments && image) {
    UpateImageRequierments(image);
  }
}

void ImageDescriptorBinding::UpateImageRequierments(
    gpu_resources::Image* image) const {
  gpu_resources::ImageProperties properties;
  properties.usage_flags |= DescriptorTypeToImageUsage(type_);
  DCHECK(image);
  image->RequireProperties(properties);
}

vk::ImageLayout ImageDescriptorBinding::GetExpectedLayout() const noexcept {
  return ImageUsageToLayout(DescriptorTypeToImageUsage(type_));
}

vk::WriteDescriptorSet ImageDescriptorBinding::GenerateWrite(
    std::vector<vk::DescriptorBufferInfo>&,
    std::vector<vk::DescriptorImageInfo>& image_info) noexcept {
  DCHECK(image_to_bind_ != nullptr) << "Can't bind null image";
  if (!image_to_bind_->GetImageView()) {
    image_to_bind_->CreateImageView();
  }
  image_info.push_back(vk::DescriptorImageInfo{
      {}, image_to_bind_->GetImageView(), GetExpectedLayout()});
  image_to_bind_ = nullptr;
  return vk::WriteDescriptorSet({}, binding_, 0, 1, type_);
}

bool ImageDescriptorBinding::IsWriteUpdateNeeded() const noexcept {
  return image_to_bind_ != nullptr;
}

}  // namespace pipeline_handler
