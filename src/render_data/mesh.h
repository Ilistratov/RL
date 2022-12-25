#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include "gpu_resources/buffer.h"

namespace render_data {

struct Mesh {
  std::vector<glm::vec4> position;
  std::vector<glm::vec4> normal;
  std::vector<glm::vec2> tex_coord;
  std::vector<glm::uvec4> index;

  Mesh() = default;
  static Mesh LoadFromObj(const std::string& obj_file_path);

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;

  Mesh(Mesh&& other) noexcept;
  void operator=(Mesh&& other) noexcept;

  void Swap(Mesh& other) noexcept;

  void ReorderPrimitives(const std::vector<uint32_t>& primirive_order);
};

}  // namespace render_data
