#include "pipeline_handler/compute.h"

#include <fstream>
#include <vector>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "base/base.h"
#include "utill/error_handling.h"
#include "utill/logger.h"

namespace pipeline_handler {

Compute::Compute(const shader::Loader& loader,
                 std::vector<DescriptorSet*> descriptor_sets,
                 const std::string& entry_point)
    : descriptor_sets_(descriptor_sets) {
  std::vector<vk::DescriptorSetLayout> vk_layouts(descriptor_sets_.size());
  for (uint32_t i = 0; i < descriptor_sets_.size(); i++) {
    DCHECK(descriptor_sets_[i]) << "Set with index " << i << " was nullptr";
    vk_layouts[i] = descriptor_sets_[i]->GetLayout();
  }
  std::vector<vk::PushConstantRange> push_constants =
      loader.GeneratePushConstantRanges();
  auto device = base::Base::Get().GetContext().GetDevice();

  layout_ = device.createPipelineLayout(
      vk::PipelineLayoutCreateInfo({}, vk_layouts, push_constants));
  auto pipeline_create_res = device.createComputePipeline(
      {}, vk::ComputePipelineCreateInfo(
              {},
              vk::PipelineShaderStageCreateInfo(
                  {}, vk::ShaderStageFlagBits::eCompute,
                  loader.GetShaderModule(), entry_point.c_str(), {}),
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
  descriptor_sets_.swap(other.descriptor_sets_);
}

void Compute::RecordDispatch(vk::CommandBuffer& cmd,
                             uint32_t group_count_x,
                             uint32_t group_count_y,
                             uint32_t group_count_z) {
  std::vector<vk::DescriptorSet> sets_to_bind;
  sets_to_bind.reserve(descriptor_sets_.size());
  for (auto set : descriptor_sets_) {
    set->SubmitUpdatesIfNeed();
    sets_to_bind.push_back(set->GetSet());
  }

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout_, 0,
                         sets_to_bind, {});
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
