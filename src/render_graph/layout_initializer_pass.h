#pragma once

#include <vector>
#include "gpu_resources/image.h"
#include "render_graph/pass.h"

namespace render_graph {

class LayoutInitializerPass : public Pass {
  std::vector<gpu_resources::Image*> preloaded_;
  std::vector<gpu_resources::Image*> dont_care_;

 public:
  LayoutInitializerPass() = default;
  LayoutInitializerPass(const std::vector<gpu_resources::Image*>& preloaded,
                        const std::vector<gpu_resources::Image*>& dont_care);

  void OnPreRecord() override;
};

}  // namespace render_graph