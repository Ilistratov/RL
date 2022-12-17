#include "pipeline_handler/descriptor_set.h"

#include <map>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"

#include "utill/error_handling.h"

namespace pipeline_handler {

DescriptorSet::DescriptorSet(const std::vector<DescriptorBinding*>& bindings)
    : bindings_(bindings) {
  std::vector<vk::DescriptorSetLayoutBinding> vk_bindings(bindings.size());
  for (uint32_t binding_ind = 0; binding_ind < bindings.size(); binding_ind++) {
    vk_bindings[binding_ind] = bindings[binding_ind]->GetVkBinding();
    vk_bindings[binding_ind].binding = binding_ind;
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
  std::swap(set_, other.set_);
  std::swap(layout_, other.layout_);
}

vk::DescriptorSetLayout DescriptorSet::GetLayout() const {
  return layout_;
}

vk::DescriptorSet DescriptorSet::GetSet() const {
  return set_;
}

void DescriptorSet::SubmitUpdatesIfNeed() {
  DCHECK(set_) << "Descriptor set must be created to use this method";
  std::vector<vk::DescriptorBufferInfo> buffer_info;
  std::vector<uint32_t> buffer_info_offset(1, 0);
  std::vector<vk::DescriptorImageInfo> image_info;
  std::vector<uint32_t> image_info_offset(1, 0);
  std::vector<vk::WriteDescriptorSet> vk_writes;

  for (uint32_t i = 0; i < bindings_.size(); i++) {
    if (!bindings_[i]->IsWriteUpdateNeeded()) {
      continue;
    }
    uint32_t current_buffer_info_offset = buffer_info_offset.back();
    uint32_t current_image_info_offset = image_info_offset.back();
    vk::WriteDescriptorSet vk_write = bindings_[i]->GenerateWrite(
        buffer_info, buffer_info_offset, image_info, image_info_offset);
    vk_write.pBufferInfo = buffer_info.data() + current_buffer_info_offset;
    vk_write.pImageInfo = image_info.data() + current_image_info_offset;
    vk_write.dstBinding = i;
    vk_write.dstSet = set_;
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.updateDescriptorSets(vk_writes, {});
}

DescriptorSet::~DescriptorSet() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorSetLayout(layout_);
}

}  // namespace pipeline_handler
