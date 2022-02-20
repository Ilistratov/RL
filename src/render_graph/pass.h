#pragma once

#include <map>
#include <string>

#include "gpu_executer/task.h"
#include "render_graph/buffer_pass_bind.h"
#include "render_graph/image_pass_bind.h"
#include "render_graph/resource_manager.h"

namespace render_graph {

class Pass : public gpu_executer::Task {
  uint32_t user_ind_ = -1;

 protected:
  std::map<std::string, BufferPassBind> buffer_binds_;
  std::map<std::string, ImagePassBind> image_binds_;

  // void AddBuffer(...)
  // void AddImage(...)

 public:
  Pass();

  void BindResources(uint32_t user_ind, ResourceManager& resource_manager);
};

}  // namespace render_graph
