#include "render_graph/pass.h"

namespace render_graph {

void Pass::BindResources(uint32_t user_ind, ResourceManager& resource_manager) {
  assert(user_ind_ == (uint32_t)(-1));
  user_ind_ = user_ind;
  for (auto& [name, buffer] : buffer_binds_) {
    buffer.OnResourceBind(user_ind_, &resource_manager.GetBuffer(name));
  }
  for (auto& [name, image] : image_binds_) {
    image.OnResourceBind(user_ind_, &resource_manager.GetImage(name));
  }
}

}  // namespace render_graph
