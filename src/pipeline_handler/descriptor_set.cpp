#include "pipeline_handler/descriptor_set.h"

#include <map>

#include "base/base.h"

namespace pipeline_handler {

DescriptorSet::DescriptorSet(
    const std::vector<const DescriptorBinding*>& bindings) {
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

void DescriptorSet::UpdateDescriptorSet(
    const std::vector<const DescriptorBinding*>& bindings) const {
  assert(set_);
  std::vector<Write> writes(bindings.size());
  std::vector<vk::WriteDescriptorSet> vk_writes(bindings.size());
  for (uint32_t i = 0; i < bindings.size(); i++) {
    writes[i] = bindings[i]->GetWrite();
    vk_writes[i] = writes[i].ConvertToVkWrite(set_, i);
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.updateDescriptorSets(vk_writes, {});
}

DescriptorSet::~DescriptorSet() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorSetLayout(layout_);
}

}  // namespace pipeline_handler
