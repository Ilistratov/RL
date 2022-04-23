#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include "gpu_resources/logical_buffer.h"

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

  vk::DeviceSize LoadToStagingBuffer(
      gpu_resources::LogicalBuffer* staging_buffer,
      vk::DeviceSize dst_offset);
  vk::DeviceSize RecordCopyFromStaging(
      vk::CommandBuffer cmd,
      gpu_resources::LogicalBuffer* staging_buffer,
      gpu_resources::LogicalBuffer* position_buffer,
      gpu_resources::LogicalBuffer* normal_buffer,
      gpu_resources::LogicalBuffer* tex_coord_buffer,
      gpu_resources::LogicalBuffer* index_buffer,
      vk::DeviceSize src_offset);
};

}  // namespace render_data
