#include "pipeline_handler/compute.h"

#include <fstream>

#include "base/base.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace pipeline_handler {

namespace {

vk::UniqueShaderModule LoadShaderModule(const std::string& file_path) {
  DLOG << "Loading shader module from " << file_path;
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    LOG << "Failed to open " << file_path;
    return {};
  }
  size_t file_size = file.tellg();
  std::vector<char> shader_binary_data(file_size);
  file.seekg(0);
  file.read(shader_binary_data.data(), file_size);
  file.close();
  auto device = base::Base::Get().GetContext().GetDevice();
  return device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(
      vk::ShaderModuleCreateFlags{}, shader_binary_data.size(),
      (const uint32_t*)shader_binary_data.data()));
}

}  // namespace

Compute::Compute(const std::vector<DescriptorBinding*>& bindings,
                 DescriptorPool& descriptor_pool,
                 const std::vector<vk::PushConstantRange>& push_constants,
                 const std::string& shader_file_path,
                 const std::string& shader_main) {
  auto device = base::Base::Get().GetContext().GetDevice();
  descriptor_set_ = descriptor_pool.ReserveDescriptorSet(bindings);
  DCHECK(descriptor_set_) << "Failed to reserve descriptor set";
  std::vector<vk::DescriptorSetLayout> vk_layouts = {
      descriptor_set_->GetLayout()};
  layout_ = device.createPipelineLayout(
      vk::PipelineLayoutCreateInfo({}, vk_layouts, push_constants));
  vk::UniqueShaderModule shader_module = LoadShaderModule(shader_file_path);
  DLOG << "Creating compute pipeline for " << shader_file_path;
  auto pipeline_create_res = device.createComputePipeline(
      {}, vk::ComputePipelineCreateInfo(
              {},
              vk::PipelineShaderStageCreateInfo(
                  {}, vk::ShaderStageFlagBits::eCompute, shader_module.get(),
                  shader_main.c_str(), {}),
              layout_));
  CHECK_VK_RESULT(pipeline_create_res.result) << "Failed to create pipeline";
  pipeline_ = pipeline_create_res.value;
}

Compute::Compute(Compute&& other) noexcept {
  Swap(other);
}

void Compute::operator=(Compute&& other) noexcept {
  Compute tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Compute::Swap(Compute& other) noexcept {
  std::swap(pipeline_, other.pipeline_);
  std::swap(layout_, other.layout_);
  std::swap(descriptor_set_, other.descriptor_set_);
}

void Compute::RecordDispatch(vk::CommandBuffer& cmd,
                             uint32_t group_count_x,
                             uint32_t group_count_y,
                             uint32_t group_count_z) {
  descriptor_set_->SubmitUpdatesIfNeed();
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout_, 0,
                         descriptor_set_->GetSet(), {});
  cmd.dispatch(group_count_x, group_count_y, group_count_z);
}

vk::PipelineLayout Compute::GetLayout() const {
  return layout_;
}

Compute::~Compute() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyPipeline(pipeline_);
  device.destroyPipelineLayout(layout_);
}

}  // namespace pipeline_handler
