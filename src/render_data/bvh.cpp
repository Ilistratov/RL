#include "render_data/bvh.h"

#include "utill/logger.h"
#include <algorithm>
#include <glm/fwd.hpp>
#include <stdint.h>


namespace render_data {

static void InterseptRange(glm::vec2 &a, const glm::vec2 &b) {
  a.x = std::max(a.x, b.x);
  a.y = std::min(a.y, b.y);
}

static void UniteRange(glm::vec2 &a, const glm::vec2 &b) {
  a.x = std::min(a.x, b.x);
  a.y = std::max(a.y, b.y);
}

static bool containsRange(const glm::vec2 &a, const glm::vec2 &b) {
  return a.x <= b.x && b.y <= a.y;
}

BoundingBox &BoundingBox::Intersept(const BoundingBox &other) {
  InterseptRange(x_range, other.x_range);
  InterseptRange(y_range, other.y_range);
  InterseptRange(z_range, other.z_range);

  return *this;
}

BoundingBox BoundingBox::GetInterseption(BoundingBox other) const {
  return other.Intersept(*this);
}

BoundingBox &BoundingBox::Unite(const BoundingBox &other) {
  UniteRange(x_range, other.x_range);
  UniteRange(y_range, other.y_range);
  UniteRange(z_range, other.z_range);

  return *this;
}

BoundingBox &BoundingBox::Unite(const glm::vec3 &pt) {
  return Unite(BoundingBox{{pt.x, pt.x}, {pt.y, pt.y}, {pt.z, pt.z}});
}

BoundingBox BoundingBox::GetUnion(BoundingBox other) const {
  return other.Unite(*this);
}

glm::vec3 BoundingBox::GetSize() const {
  return glm::vec3(std::max(x_range.y - x_range.x, 0.0f),
                   std::max(y_range.y - y_range.x, 0.0f),
                   std::max(z_range.y - z_range.x, 0.0f));
}

float BoundingBox::GetVolume() const {
  glm::vec3 size = GetSize();
  return size.x * size.y * size.z;
}

float BoundingBox::GetSurfaceArea() const {
  glm::vec3 size = GetSize();
  return (size.x * size.y + size.y * size.z + size.z * size.x) * 2;
}

bool BoundingBox::IsEmpty() const {
  glm::vec3 size = GetSize();
  return size.x == 0 || size.y == 0 || size.z == 0;
}

bool BoundingBox::Contains(const BoundingBox &other) const {
  return containsRange(x_range, other.x_range) &&
         containsRange(y_range, other.y_range) &&
         containsRange(z_range, other.z_range);
}

void BVH::OrderPrimitives(
    std::vector<std::pair<BoundingBox, uint32_t>> &primitives, uint32_t v,
    uint32_t l, uint32_t r) {
  glm::vec3 bounds = node_[v].bounds.GetSize();
  if (bounds.x > std::max(bounds.y, bounds.z)) {
    std::sort(primitives.begin() + l, primitives.begin() + r,
              [](const std::pair<BoundingBox, uint32_t> &lhs,
                 const std::pair<BoundingBox, uint32_t> &rhs) {
                return lhs.first.x_range.x < rhs.first.x_range.x;
              });
  } else if (bounds.y > std::max(bounds.x, bounds.z)) {
    std::sort(primitives.begin() + l, primitives.begin() + r,
              [](const std::pair<BoundingBox, uint32_t> &lhs,
                 const std::pair<BoundingBox, uint32_t> &rhs) {
                return lhs.first.y_range.x < rhs.first.y_range.x;
              });
  } else {
    std::sort(primitives.begin() + l, primitives.begin() + r,
              [](const std::pair<BoundingBox, uint32_t> &lhs,
                 const std::pair<BoundingBox, uint32_t> &rhs) {
                return lhs.first.z_range.x < rhs.first.z_range.x;
              });
  }
}

BVH::NodeSeparation BVH::CalcOptimalSeparation(
    std::vector<std::pair<BoundingBox, uint32_t>> &primitives, uint32_t v,
    uint32_t l, uint32_t r) {
  bb_pool_[r - 1] = primitives[r - 1].first;

  for (uint32_t i = r - 1; i > l; i--) {
    bb_pool_[i - 1] = bb_pool_[i];
    bb_pool_[i - 1].Unite(primitives[i - 1].first);
  }

  BoundingBox left_bb;
  NodeSeparation res;
  float cur_surf_area = node_[v].bounds.GetSurfaceArea();
  res.sep_ind = l;
  res.cost = r - l;

  for (uint32_t i = l; i + 1 < r; i++) {
    left_bb.Unite(primitives[i].first);
    float cur_cost =
        (i - l + 1) * (left_bb.GetSurfaceArea() / cur_surf_area) +
        (r - i - 1) * (bb_pool_[i + 1].GetSurfaceArea() / cur_surf_area);

    if (cur_cost < res.cost) {
      res.left_bb = left_bb;
      res.right_bb = bb_pool_[i + 1];
      res.cost = cur_cost;
      res.sep_ind = i + 1;
    }
  }
  return res;
}

void BVH::MakeLeaf(std::vector<std::pair<BoundingBox, uint32_t>> &primitives,
                   uint32_t v, uint32_t l, uint32_t r) {
  node_[v].left = l;
  node_[v].right = r;
  node_[v].bvh_level = (int32_t)(-1);
  for (uint32_t i = l; i < r; i++) {
    primitive_ord_[i] = primitives[i].second;
  }
}

void BVH::Construct(std::vector<std::pair<BoundingBox, uint32_t>> &primitives,
                    uint32_t v, uint32_t l, uint32_t r, uint32_t d,
                    uint32_t max_d, uint32_t min_node_primitives) {
  if (r - l <= min_node_primitives || d == max_d) {
    MakeLeaf(primitives, v, l, r);
    return;
  }

  OrderPrimitives(primitives, v, l, r);
  NodeSeparation sep = CalcOptimalSeparation(primitives, v, l, r);

  if (sep.cost + 0.5 > r - l) {
    MakeLeaf(primitives, v, l, r);
    return;
  }

  node_[v].left = node_.size();
  node_.push_back(BVHNode{sep.left_bb, 0, 0, v, node_[v].bvh_level + 1u});
  Construct(primitives, node_[v].left, l, sep.sep_ind, d + 1, max_d,
            min_node_primitives);

  node_[v].right = node_.size();
  node_.push_back(BVHNode{sep.right_bb, 0, 0, v, node_[v].bvh_level + 1u});
  Construct(primitives, node_[v].right, sep.sep_ind, r, d + 1, max_d,
            min_node_primitives);
}

static inline void PrintBvhStats(std::vector<BVHNode> const &nodes) {
  uint32_t max_depth = 0;
  uint32_t leaf_count = 0;
  uint32_t primitive_count = 0;
  for (BVHNode const &node : nodes) {
    if (node.bvh_level == (uint32_t)-1) {
      ++leaf_count;
      primitive_count += node.right - node.left;
    } else {
      max_depth = std::max(max_depth, node.bvh_level);
    }
  }
  LOG << "Constructed BVH. Nodes: " << nodes.size()
      << " Max depth: " << max_depth << " Leafs: " << leaf_count
      << " Avg primitives per leaf: " << (double)primitive_count / leaf_count;
}

BVH::BVH(std::vector<std::pair<BoundingBox, uint32_t>> &&primitives,
         uint32_t max_depth, uint32_t min_node_primitives) {
  primitive_ord_.resize(primitives.size());
  bb_pool_.resize(primitives.size() + 1);
  node_.reserve(primitives.size() * 2 - 1);
  node_.push_back({});
  for (const auto &[pbb, ind] : primitives) {
    node_[0].bounds.Unite(pbb);
  }
  Construct(primitives, 0, 0, primitives.size(), 0, max_depth,
            min_node_primitives);
  bb_pool_.clear();
  bb_pool_.shrink_to_fit();
  node_.shrink_to_fit();
  PrintBvhStats(node_);
}

std::vector<std::pair<BoundingBox, uint32_t>>
BVH::BuildPrimitivesBB(const Mesh &mesh) {
  std::vector<std::pair<BoundingBox, uint32_t>> res(mesh.index.size() / 3);
  for (uint32_t i = 0; i * 3 < mesh.index.size(); i++) {
    res[i].first.Unite(mesh.position[mesh.index[3 * i + 0].x]);
    res[i].first.Unite(mesh.position[mesh.index[3 * i + 1].x]);
    res[i].first.Unite(mesh.position[mesh.index[3 * i + 2].x]);
    res[i].second = i;
  }
  return res;
}

const std::vector<BVHNode> &BVH::GetNodes() const { return node_; }

const std::vector<uint32_t> &BVH::GetPrimitiveOrd() const {
  return primitive_ord_;
}

} // namespace render_data
