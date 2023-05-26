#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/buffer.h"
#include "gpu_resources/image.h"
#include "pipeline_handler/descriptor_binding.h"

namespace pipeline_handler {

class DescriptorSet {
  std::vector<BufferDescriptorBinding> buffer_bindings_;
  std::vector<ImageDescriptorBinding> image_bindings_;
  vk::DescriptorSetLayout layout_ = {};
  vk::DescriptorSet set_ = {};
  uint32_t set_idx_ = 0;

  friend class DescriptorPool;

  DescriptorSet(uint32_t set_idx,
                std::vector<BufferDescriptorBinding> buffer_bindings,
                std::vector<ImageDescriptorBinding> image_bindings);

 public:
  DescriptorSet() = default;
  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&& other) noexcept;
  void operator=(DescriptorSet&& other) noexcept;
  void Swap(DescriptorSet& other) noexcept;

  vk::DescriptorSetLayout GetLayout() const;
  vk::DescriptorSet GetSet() const;

  BufferDescriptorBinding* GetBufferBinding(uint32_t binding_num);
  ImageDescriptorBinding* GetImageBinding(uint32_t binding_num);

  void BulkBind(std::vector<gpu_resources::Buffer*> buffers_to_bind,
                bool update_requirements = false);
  void BulkBind(std::vector<gpu_resources::Image*> images_to_bind,
                bool update_requirements = false);

  void SubmitUpdatesIfNeed();

  ~DescriptorSet();
};

}  // namespace pipeline_handler
