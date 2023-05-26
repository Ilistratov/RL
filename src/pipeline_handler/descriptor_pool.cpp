#include "pipeline_handler/descriptor_pool.h"

#include "base/base.h"

#include "pipeline_handler/descriptor_set.h"
#include "utill/error_handling.h"

namespace pipeline_handler {

void DescriptorPool::CreatePool() {
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

void DescriptorPool::AllocateSets() {
  std::vector<vk::DescriptorSetLayout> managed_set_layouts;
  managed_set_layouts.reserve(managed_sets_.size());
  for (auto &set : managed_sets_) {
    managed_set_layouts.push_back(set.GetLayout());
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  auto sets = device.allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo(pool_, managed_set_layouts));
  uint32_t set_ind = 0;
  for (auto &set : managed_sets_) {
    assert(!set.set_);
    set.set_ = sets[set_ind];
    ++set_ind;
  }
}

DescriptorSet *DescriptorPool::ReserveDescriptorSet(
    uint32_t set_num, std::vector<BufferDescriptorBinding> buffer_bindings,
    std::vector<ImageDescriptorBinding> image_bindings) {
  for (const auto &binding : buffer_bindings) {
    descriptor_type_reserved_count_[binding.GetType()] += binding.GetCount();
  }
  for (const auto &binding : image_bindings) {
    descriptor_type_reserved_count_[binding.GetType()] += binding.GetCount();
  }
  pipeline_handler::DescriptorSet d_set(set_num, buffer_bindings,
                                        image_bindings);
  managed_sets_.push_back(std::move(d_set));
  return &managed_sets_.back();
}

void DescriptorPool::Create() {
  DCHECK(!pool_) << "Expected null pool during creation";
  CreatePool();
  AllocateSets();
}

DescriptorPool::~DescriptorPool() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyDescriptorPool(pool_);
}

} // namespace pipeline_handler
