#pragma once

#include <list>
#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "pipeline_handler/binding.h"
#include "pipeline_handler/set.h"

namespace pipeline_handler {

class Pool {
  std::list<Set> managed_sets_;
  vk::DescriptorPool pool_;
  std::map<vk::DescriptorType, uint32_t> descriptor_type_reserved_count_;

  void CreatePool();
  void AllocateSets();

 public:
  Pool() = default;

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  Set* ReserveDescriptorSet(const std::vector<const Binding*>& bindings);
  void Create();

  ~Pool();
};

}  // namespace pipeline_handler
