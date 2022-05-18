#include "render_data/mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "utill/error_handling.h"
#include "utill/logger.h"

namespace render_data {

static std::vector<glm::vec4> UnflattenCordVec(
    const std::vector<tinyobj::real_t>& src,
    float w_comp) {
  std::vector<glm::vec4> res;
  size_t step = 3;
  res.reserve(src.size() / step);
  for (size_t i = 0; i < src.size(); i += step) {
    res.push_back(glm::vec4(src[i + 0], src[i + 1], src[i + 2], w_comp));
  }
  return res;
}

Mesh Mesh::LoadFromObj(const std::string& obj_file_path) {
  Mesh result;
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig cfg;
  cfg.triangulate = true;

  if (!reader.ParseFromFile(obj_file_path, cfg)) {
    LOG << "Failed to parse obj file: " << obj_file_path
        << " error: " << reader.Error();
    return result;
  }
  if (!reader.Warning().empty()) {
    LOG << "Warning while parsing obj file: " << obj_file_path << ". "
        << reader.Warning();
  }

  const auto& attrib = reader.GetAttrib();
  result.position = UnflattenCordVec(attrib.vertices, 1.0);
  result.normal = UnflattenCordVec(attrib.normals, 0.0);
  result.tex_coord.reserve(attrib.texcoords.size() / 2);
  for (uint32_t i = 0; i < attrib.texcoords.size(); i += 2) {
    result.tex_coord.push_back(
        glm::vec2{attrib.texcoords[i + 0], attrib.texcoords[i + 1]});
  }

  const auto& shapes = reader.GetShapes();
  uint32_t ind_size = 0;
  for (const auto& shape : shapes) {
    ind_size += shape.mesh.indices.size();
  }
  result.index.reserve(ind_size);
  for (const auto& shape : shapes) {
    for (uint32_t i = 0; i < shape.mesh.indices.size(); i++) {
      result.index.push_back(glm::uvec4{shape.mesh.indices[i].vertex_index,
                                        shape.mesh.indices[i].normal_index,
                                        shape.mesh.indices[i].texcoord_index,
                                        0});
    }
  }
  LOG << "Loaded mesh from obj: " << obj_file_path;
  return result;
}

Mesh::Mesh(Mesh&& other) noexcept {
  Mesh tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Mesh::operator=(Mesh&& other) noexcept {
  Mesh tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Mesh::Swap(Mesh& other) noexcept {
  position.swap(other.position);
  normal.swap(other.normal);
  tex_coord.swap(other.tex_coord);
  index.swap(other.index);
}

vk::DeviceSize Mesh::LoadToStagingBuffer(gpu_resources::Buffer* staging_buffer,
                                         vk::DeviceSize dst_offset) {
  DCHECK(staging_buffer) << "Staging buffer must be presented";
  dst_offset = staging_buffer->LoadDataFromVec(position, dst_offset);
  DCHECK(dst_offset != (vk::DeviceSize)(-1)) << "Failed to load position";
  dst_offset = staging_buffer->LoadDataFromVec(normal, dst_offset);
  DCHECK(dst_offset != (vk::DeviceSize)(-1)) << "Failed to load normal";
  dst_offset = staging_buffer->LoadDataFromVec(tex_coord, dst_offset);
  DCHECK(dst_offset != (vk::DeviceSize)(-1)) << "Failed to load tex_coord";
  dst_offset = staging_buffer->LoadDataFromVec(index, dst_offset);
  DCHECK(dst_offset != (vk::DeviceSize)(-1)) << "Failed to load index";
  return dst_offset;
}

static vk::DeviceSize RecordCopy(vk::CommandBuffer cmd,
                                 gpu_resources::Buffer* staging_buffer,
                                 gpu_resources::Buffer* dst_buffer,
                                 vk::DeviceSize data_size,
                                 vk::DeviceSize src_offset) {
  if (!dst_buffer || data_size == 0) {
    return src_offset;
  }
  gpu_resources::Buffer::RecordCopy(cmd, *staging_buffer, *dst_buffer,
                                    src_offset, 0, data_size);
  return src_offset + data_size;
}

vk::DeviceSize Mesh::RecordCopyFromStaging(
    vk::CommandBuffer cmd,
    gpu_resources::Buffer* staging_buffer,
    gpu_resources::Buffer* position_buffer,
    gpu_resources::Buffer* normal_buffer,
    gpu_resources::Buffer* tex_coord_buffer,
    gpu_resources::Buffer* index_buffer,
    vk::DeviceSize src_offset) {
  using gpu_resources::GetDataSize;
  DCHECK(staging_buffer) << "Staging buffer must be presented";
  src_offset = RecordCopy(cmd, staging_buffer, position_buffer,
                          GetDataSize(position), src_offset);
  src_offset = RecordCopy(cmd, staging_buffer, normal_buffer,
                          GetDataSize(normal), src_offset);
  src_offset = RecordCopy(cmd, staging_buffer, tex_coord_buffer,
                          GetDataSize(tex_coord), src_offset);
  src_offset = RecordCopy(cmd, staging_buffer, index_buffer, GetDataSize(index),
                          src_offset);
  return src_offset;
}

void Mesh::ReorderPrimitives(const std::vector<uint32_t>& primirive_order) {
  std::vector<glm::uvec4> n_index;
  n_index.reserve(index.size());
  for (uint32_t n_ind : primirive_order) {
    n_index.push_back(index[3 * n_ind + 0]);
    n_index.push_back(index[3 * n_ind + 1]);
    n_index.push_back(index[3 * n_ind + 2]);
  }
  index.swap(n_index);
}

}  // namespace render_data
