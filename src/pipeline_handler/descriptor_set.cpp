#include "pipeline_handler/descriptor_set.h"

#include <stdint.h>
#include <algorithm>
#include <map>
#include <vector>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"

#include "pipeline_handler/descriptor_binding.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace pipeline_handler {

struct BindingCmp {
  bool operator()(const DescriptorBinding& lhs,
                  const DescriptorBinding& rhs) const noexcept {
    return lhs.GetNum() < rhs.GetNum();
  }

  bool operator()(const DescriptorBinding& lhs, uint32_t num) const noexcept {
    return lhs.GetNum() < num;
  }
};

DescriptorSet::DescriptorSet(
    uint32_t set_idx,
    std::vector<BufferDescriptorBinding> buffer_bindings,
    std::vector<ImageDescriptorBinding> image_bindings)
    : buffer_bindings_(buffer_bindings),
      image_bindings_(image_bindings),
      set_idx_(set_idx) {
  std::sort(buffer_bindings_.begin(), buffer_bindings_.end(), BindingCmp{});
  std::sort(image_bindings_.begin(), image_bindings_.end(), BindingCmp{});
  std::vector<vk::DescriptorSetLayoutBinding> vk_bindings;
  vk_bindings.reserve(buffer_bindings_.size() + image_bindings_.size());
  for (const auto& binding : buffer_bindings) {
    vk_bindings.push_back(binding.GetVkBinding());
  }
  for (const auto& binding : image_bindings) {
    vk_bindings.push_back(binding.GetVkBinding());
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  layout_ = device.createDescriptorSetLayout(
      vk::DescriptorSetLayoutCreateInfo({}, vk_bindings));
}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept {
  Swap(other);
}
void DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  DescriptorSet tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void DescriptorSet::Swap(DescriptorSet& other) noexcept {
  buffer_bindings_.swap(other.buffer_bindings_);
  image_bindings_.swap(other.image_bindings_);
  std::swap(layout_, other.layout_);
  std::swap(set_, other.set_);
  std::swap(set_idx_, other.set_idx_);
}

vk::DescriptorSetLayout DescriptorSet::GetLayout() const {
  return layout_;
}

vk::DescriptorSet DescriptorSet::GetSet() const {
  return set_;
}

BufferDescriptorBinding* DescriptorSet::GetBufferBinding(uint32_t binding_num) {
  auto it = std::lower_bound(buffer_bindings_.begin(), buffer_bindings_.end(),
                             binding_num, BindingCmp{});
  if (it == buffer_bindings_.end() || (*it).GetNum() != binding_num) {
    return nullptr;
  }
  return &(*it);
}

ImageDescriptorBinding* DescriptorSet::GetImageBinding(uint32_t binding_num) {
  auto it = std::lower_bound(image_bindings_.begin(), image_bindings_.end(),
                             binding_num, BindingCmp{});
  if (it == image_bindings_.end() || (*it).GetNum() != binding_num) {
    return nullptr;
  }
  return &(*it);
}

void DescriptorSet::BulkBind(
    std::vector<gpu_resources::Buffer*> buffers_to_bind,
    bool update_requirements) {
  for (uint32_t i = 0;
       i < buffers_to_bind.size() && i < buffer_bindings_.size(); i++) {
    if (buffers_to_bind[i]) {
      DLOG << "Binding buffer from set " << set_idx_ << " binding "
           << buffer_bindings_[i].GetNum();
      buffer_bindings_[i].SetBuffer(buffers_to_bind[i], update_requirements);
    }
  }
}

void DescriptorSet::BulkBind(std::vector<gpu_resources::Image*> images_to_bind,
                             bool update_requirements) {
  for (uint32_t i = 0; i < images_to_bind.size() && i < image_bindings_.size();
       i++) {
    if (images_to_bind[i]) {
      image_bindings_[i].SetImage(images_to_bind[i], update_requirements);
    }
  }
}

void DescriptorSet::SubmitUpdatesIfNeed() {
  DCHECK(set_) << "Descriptor set must be created to use this method";
  std::vector<vk::DescriptorBufferInfo> buffer_info;
  std::vector<vk::DescriptorImageInfo> image_info;
  std::vector<vk::WriteDescriptorSet> vk_writes;
  std::vector<DescriptorBinding*> bindings_to_update;

  for (auto& binding : buffer_bindings_) {
    if (binding.IsWriteUpdateNeeded()) {
      bindings_to_update.push_back(&binding);
    }
  }
  for (auto& binding : image_bindings_) {
    if (binding.IsWriteUpdateNeeded()) {
      bindings_to_update.push_back(&binding);
    }
  }

  for (auto binding : bindings_to_update) {
    vk::WriteDescriptorSet vk_write =
        binding->GenerateWrite(buffer_info, image_info);
    vk_write.dstSet = set_;
    vk_writes.push_back(vk_write);
  }
  uint32_t buffer_info_offset = 0;
  uint32_t image_info_offset = 0;
  for (uint32_t i = 0; i < vk_writes.size(); i++) {
    if (bindings_to_update[i]->IsImageBinding()) {
      vk_writes[i].pImageInfo = image_info.data() + image_info_offset;
      image_info_offset += bindings_to_update[i]->GetCount();
    } else {
      vk_writes[i].pBufferInfo = buffer_info.data() + buffer_info_offset;
      buffer_info_offset += bindings_to_update[i]->GetCount();
    }
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.updateDescriptorSets(vk_writes, {});
}

DescriptorSet::~DescriptorSet() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorSetLayout(layout_);
}

}  // namespace pipeline_handler
