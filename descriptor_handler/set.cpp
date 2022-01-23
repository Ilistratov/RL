#include "descriptor_handler/set.h"

#include <map>

#include "base/base.h"

namespace descriptor_handler {

Set::Set(const std::vector<const Binding*>& bindings) {
  std::vector<vk::DescriptorSetLayoutBinding> vk_bindings(bindings.size());
  for (uint32_t binding_ind = 0; binding_ind < bindings.size(); binding_ind++) {
    vk_bindings[binding_ind] = bindings[binding_ind]->GetVkBinding();
    vk_bindings[binding_ind].binding = binding_ind;
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  layout_ = device.createDescriptorSetLayout(
      vk::DescriptorSetLayoutCreateInfo({}, vk_bindings));
}

vk::DescriptorSetLayout Set::GetLayout() const {
  return layout_;
}

vk::DescriptorSet Set::GetSet() const {
  return set_;
}

void Set::UpdateDescriptorSet(
    const std::vector<const Binding*>& bindings) const {
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

Set::~Set() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorSetLayout(layout_);
}

}  // namespace descriptor_handler
