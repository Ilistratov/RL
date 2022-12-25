#include "render_graph/layout_initializer_pass.h"
#include <vulkan/vulkan_enums.hpp>
#include "gpu_resources/resource_access_syncronizer.h"

namespace render_graph {

LayoutInitializerPass::LayoutInitializerPass(
    const std::vector<gpu_resources::Image*>& preloaded,
    const std::vector<gpu_resources::Image*>& dont_care)
    : preloaded_(preloaded), dont_care_(dont_care) {}

void LayoutInitializerPass::OnPreRecord() {
  gpu_resources::ResourceAccess preloaded_access{};
  preloaded_access.layout = vk::ImageLayout::ePreinitialized;
  preloaded_access.stage_flags = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
  for (auto image : preloaded_) {
    image->DeclareAccess(preloaded_access, GetPassIdx());
  }
  preloaded_.clear();

  gpu_resources::ResourceAccess dont_care_access{};
  dont_care_access.layout = vk::ImageLayout::eUndefined;
  dont_care_access.stage_flags = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
  for (auto image : dont_care_) {
    image->DeclareAccess(dont_care_access, GetPassIdx());
  }
  // don't clear as we wan to reset it every time
}

}  // namespace render_graph