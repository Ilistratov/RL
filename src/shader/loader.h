#pragma once

#include <stdint.h>
#include <spirv_cross/spirv_reflect.hpp>
#include <string>
#include <vector>

#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_cross_containers.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_pool.h"
#include "pipeline_handler/descriptor_set.h"

namespace shader {

class Loader {
  spirv_cross::CompilerReflection compiler_;
  vk::ShaderStageFlags shader_stage_;
  vk::ShaderModule shader_module_;

  uint32_t GetSet(const spirv_cross::Resource& resource) const noexcept;
  uint32_t GetBinding(const spirv_cross::Resource& resource) const noexcept;

  template <typename T>
  void PopulateBindings(
      const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
      std::vector<T>& bindings,
      uint32_t set,
      vk::DescriptorType descriptor_type) const;

 public:
  Loader(const char* path);
  Loader(const std::vector<uint32_t>& spirv_binary);

  pipeline_handler::DescriptorSet* GenerateDescriptorSet(
      pipeline_handler::DescriptorPool& pool,
      uint32_t set) const;

  std::vector<vk::PushConstantRange> GeneratePushConstantRanges() const;

  vk::ShaderStageFlags GetShaderStage() const;
  vk::ShaderModule GetShaderModule() const;

  Loader(const Loader&) = delete;
  Loader& operator=(const Loader&) = delete;

  ~Loader();
};

template <typename T>
void Loader::PopulateBindings(
    const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
    std::vector<T>& bindings,
    uint32_t set,
    vk::DescriptorType descriptor_type) const {
  static_assert(std::is_base_of_v<pipeline_handler::DescriptorBinding, T>,
                "T must be descriptor binding");
  for (auto resource : resources) {
    if (GetSet(resource) != set) {
      continue;
    }
    uint32_t binding = GetBinding(resource);
    bindings.push_back(T(descriptor_type, binding, shader_stage_));
  }
}

}  // namespace shader