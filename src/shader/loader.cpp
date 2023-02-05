#include "shader/loader.h"

#include <stdint.h>
#include <fstream>
#include <vector>

#include <spirv_cross/spirv.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_cross_parsed_ir.hpp>
#include <spirv_cross/spirv_reflect.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "pipeline_handler/descriptor_binding.h"
#include "pipeline_handler/descriptor_set.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace shader {

namespace {

std::vector<uint32_t> LoadSpirvIR(const char* path) {
  DLOG << "Reading spirv IR from " << path;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    LOG << "Failed to open " << path;
    return {};
  }
  size_t file_size = file.tellg();
  DCHECK(file_size % sizeof(uint32_t) == 0);
  std::vector<uint32_t> shader_binary(file_size / sizeof(uint32_t));
  file.seekg(0);
  file.read((char*)shader_binary.data(), file_size);
  file.close();

  return shader_binary;
}

}  // namespace

uint32_t Loader::GetSet(const spirv_cross::Resource& resource) const noexcept {
  return compiler_.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

uint32_t Loader::GetBinding(
    const spirv_cross::Resource& resource) const noexcept {
  return compiler_.get_decoration(resource.id, spv::DecorationBinding);
}

Loader::Loader(const char* path) : Loader(LoadSpirvIR(path)) {}

Loader::Loader(const std::vector<uint32_t>& spirv_binary)
    : compiler_(spirv_binary) {
  switch (compiler_.get_execution_model()) {
    case spv::ExecutionModelGLCompute:
      shader_stage_ = vk::ShaderStageFlagBits::eCompute;
      break;
    case spv::ExecutionModelVertex:
    case spv::ExecutionModelTessellationControl:
    case spv::ExecutionModelTessellationEvaluation:
    case spv::ExecutionModelGeometry:
    case spv::ExecutionModelFragment:
    case spv::ExecutionModelKernel:
    case spv::ExecutionModelTaskNV:
    case spv::ExecutionModelMeshNV:
    case spv::ExecutionModelRayGenerationKHR:
    case spv::ExecutionModelIntersectionKHR:
    case spv::ExecutionModelAnyHitKHR:
    case spv::ExecutionModelClosestHitKHR:
    case spv::ExecutionModelMissKHR:
    case spv::ExecutionModelCallableKHR:
    case spv::ExecutionModelTaskEXT:
    case spv::ExecutionModelMeshEXT:
    case spv::ExecutionModelMax:
    default:
      CHECK(false) << "Unsupported execution model "
                   << compiler_.get_execution_model();
  }
  auto device = base::Base::Get().GetContext().GetDevice();
  shader_module_ = device.createShaderModule(
      vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags{}, spirv_binary));
  DCHECK(shader_module_) << "Failed to create shader module";
}

pipeline_handler::DescriptorSet* Loader::GenerateDescriptorSet(
    pipeline_handler::DescriptorPool& pool,
    uint32_t set) const {
  std::vector<pipeline_handler::BufferDescriptorBinding> buffer_bindings;
  std::vector<pipeline_handler::ImageDescriptorBinding> image_bindings;
  const auto shader_resources = compiler_.get_shader_resources();

  PopulateBindings(shader_resources.storage_buffers, buffer_bindings, set,
                   vk::DescriptorType::eStorageBuffer);
  PopulateBindings(shader_resources.uniform_buffers, buffer_bindings, set,
                   vk::DescriptorType::eUniformBuffer);
  PopulateBindings(shader_resources.storage_images, image_bindings, set,
                   vk::DescriptorType::eStorageImage);
  // TODO Add more as they are supported
  return pool.ReserveDescriptorSet(set, buffer_bindings, image_bindings);
}

std::vector<vk::PushConstantRange> Loader::GeneratePushConstantRanges() const {
  const auto push_constants =
      compiler_.get_shader_resources().push_constant_buffers;
  std::vector<vk::PushConstantRange> result;
  result.reserve(push_constants.size());
  for (auto push_constant : push_constants) {
    // or base_type_id ?
    auto push_constant_type = compiler_.get_type(push_constant.type_id);
    size_t pc_size = compiler_.get_declared_struct_size(push_constant_type);
    result.push_back(vk::PushConstantRange(shader_stage_, 0, pc_size));
  }
  return result;
}

vk::ShaderStageFlags Loader::GetShaderStage() const {
  return shader_stage_;
}

vk::ShaderModule Loader::GetShaderModule() const {
  return shader_module_;
}

Loader::~Loader() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyShaderModule(shader_module_);
}

}  // namespace shader
