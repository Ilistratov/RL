#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "render_data/mesh.h"

namespace render_data {

struct BoundingBox {
  glm::vec2 x_range = {1e9, -1e9};
  glm::vec2 y_range = {1e9, -1e9};
  glm::vec2 z_range = {1e9, -1e9};

  BoundingBox& Intersept(const BoundingBox& other);
  BoundingBox GetInterseption(BoundingBox other) const;

  BoundingBox& Unite(const BoundingBox& other);
  BoundingBox& Unite(const glm::vec3& pt);
  BoundingBox GetUnion(BoundingBox other) const;

  glm::vec3 GetSize() const;
  float GetVolume() const;
  bool IsEmpty() const;
  bool Contains(const BoundingBox& other) const;
};

struct BVHNode {
  BoundingBox bounds;
  uint32_t left = uint32_t(-1);
  uint32_t right = uint32_t(-1);
  uint32_t parent = 0;
  uint32_t bvh_level = 0;
};

class BVH {
  std::vector<BVHNode> node_;
  std::vector<uint32_t> primitive_ord_;
  std::vector<BoundingBox> bb_pool_;

  void OrderPrimitives(
      std::vector<std::pair<BoundingBox, uint32_t>>& primitives,
      uint32_t v,
      uint32_t l,
      uint32_t r);

  struct NodeSeparation {
    BoundingBox left_bb;
    BoundingBox right_bb;
    uint32_t sep_ind = 0;
    float cost = 0;
  };

  NodeSeparation CalcOptimalSeparation(
      std::vector<std::pair<BoundingBox, uint32_t>>& primitives,
      uint32_t v,
      uint32_t l,
      uint32_t r);

  void MakeLeaf(std::vector<std::pair<BoundingBox, uint32_t>>& primitives,
                uint32_t v,
                uint32_t l,
                uint32_t r);

  void Construct(std::vector<std::pair<BoundingBox, uint32_t>>& primitives,
                 uint32_t v,
                 uint32_t l,
                 uint32_t r,
                 uint32_t d,
                 uint32_t max_d = 32,
                 uint32_t min_node_primitives = 4);

 public:
  BVH() = default;
  BVH(std::vector<std::pair<BoundingBox, uint32_t>>&& primitives);

  static std::vector<std::pair<BoundingBox, uint32_t>> BuildPrimitivesBB(
      const Mesh& mesh);

  const std::vector<BVHNode>& GetNodes() const;
  const std::vector<uint32_t>& GetPrimitiveOrd() const;
};

}  // namespace render_data
