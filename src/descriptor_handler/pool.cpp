#include "descriptor_handler/pool.h"

#include "base/base.h"

namespace descriptor_handler {

void Pool::CreatePool() {
  std::vector<vk::DescriptorPoolSize> pool_sizes;
  pool_sizes.reserve(descriptor_type_reserved_count_.size());
  for (auto [type, count] : descriptor_type_reserved_count_) {
    pool_sizes.push_back({type, count});
  }
  descriptor_type_reserved_count_.clear();
  auto device = base::Base::Get().GetContext().GetDevice();
  pool_ = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
      {}, static_cast<uint32_t>(managed_sets_.size()), pool_sizes});
}

void Pool::AllocateSets() {
  std::vector<vk::DescriptorSetLayout> managed_set_layouts;
  managed_set_layouts.reserve(managed_sets_.size());
  for (auto& set : managed_sets_) {
    managed_set_layouts.push_back(set.GetLayout());
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  auto sets = device.allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo(pool_, managed_set_layouts));
  uint32_t set_ind = 0;
  for (auto& set : managed_sets_) {
    assert(!set.set_);
    set.set_ = sets[set_ind];
    ++set_ind;
  }
}

Set* Pool::ReserveDescriptorSet(const std::vector<const Binding*>& bindings) {
  for (uint32_t binding_ind = 0; binding_ind < bindings.size(); binding_ind++) {
    auto vk_binding = bindings[binding_ind]->GetVkBinding();
    descriptor_type_reserved_count_[vk_binding.descriptorType] +=
        vk_binding.descriptorCount;
  }
  managed_sets_.emplace_back(bindings);
  return &managed_sets_.back();
}

void Pool::Create() {
  assert(!pool_);
  CreatePool();
  AllocateSets();
}

Pool::~Pool() {
  std::vector<vk::DescriptorSet> sets;
  sets.reserve(managed_sets_.size());
  for (auto& set : managed_sets_) {
    sets.push_back(set.set_);
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorPool(pool_);
}

}  // namespace descriptor_handler
