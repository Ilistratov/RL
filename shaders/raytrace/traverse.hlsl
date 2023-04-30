#ifndef RAYTRACE_TRAVERSE
#define RAYTRACE_TRAVERSE

#include "common.hlsl"
#include "bvh.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<float4> g_vertex_pos;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> g_vertex_ind;
[[vk::binding(2, 0)]] StructuredBuffer<float4> g_vertex_normal;
[[vk::binding(3, 0)]] StructuredBuffer<BVHNode> g_bvh_buffer;

struct Triangle {
  float3 a;
  float3 b;
  float3 c;

  float3 GetNormal() {
    return normalize(cross(c - a, b - a));
  }

  float3 GetPointFromBarCord(float2 cord) {
    return a + (b - a) * cord.x + (c - a) * cord.y;
  }

  //returns insp distance, uv coord's of interseption/
  float3 GetRayInterseption(Ray r) {
    float3x3 sys_mat = float3x3(-r.direction, b - a, c - a);
    float det_sys = determinant(sys_mat);
    if (det_sys < 1e-4) {
      det_sys = -1;
    }
    float3 sys_coef = r.origin - a;
    float inv_det_sys = 1.0 / det_sys;

    float t = determinant(float3x3(sys_coef, b - a, c - a)) * inv_det_sys;
    float u = determinant(float3x3(-r.direction, sys_coef, c - a)) * inv_det_sys;
    float v = determinant(float3x3(-r.direction, b - a, sys_coef)) * inv_det_sys;
    bool isValid = det_sys > 0 & t > 1e-4 & 0 < u & u < 1 & 0 < v & v < 1 & u + v < 1;
    return isValid ? float3(t, u, v) : float3(-1, 0, 0);
  }
};

Triangle GetTriangleByInd(uint ind) {
  Triangle res;
  res.a = g_vertex_pos[g_vertex_ind[3 * ind + 0].x].xyz;
  res.b = g_vertex_pos[g_vertex_ind[3 * ind + 1].x].xyz;
  res.c = g_vertex_pos[g_vertex_ind[3 * ind + 2].x].xyz;
  return res;
}

Interception CheckLeafPrimitives(Ray r, uint vrt_ind) {
  Interception res;
  res.t = -1;
  for (uint t_ind = g_bvh_buffer[vrt_ind].left; t_ind < g_bvh_buffer[vrt_ind].right; t_ind++) {
    Triangle t = GetTriangleByInd(t_ind);
    float3 n_insp = t.GetRayInterseption(r);
    if (n_insp.x > 0 && (n_insp.x < res.t || res.t < 0)) {
      res.t = n_insp.x;
      res.u = n_insp.y;
      res.v = n_insp.z;
      res.primitive_ind = t_ind;
    }
  }
  return res;
}

const static uint kTraversalOrderUndefined = 0;
const static uint kTraversalOrderNone = 1;
const static uint kTraversalOrderOnlyLeft = 2;
const static uint kTraversalOrderOnlyRight = 3;
const static uint kTraversalOrderLeftRight = 4;
const static uint kTraversalOrderRightLeft = 5;
const static uint kTraversalOrderMax = 6;

groupshared uint l_traversal_order_votes[kTraverseThreadsPerGroup];
const static uint kMaxBVHDepth = 64;
groupshared uint l_traversal_order_stack[kMaxBVHDepth];

bool IsTraversalOmittable(float2 insp_t, float cur_res) {
  return insp_t.y < 0 || insp_t.x > insp_t.y || (cur_res > 0 && cur_res + 1e-2 < insp_t.x);
}

uint CalcTraversalOrderVote(Ray r, float current_insp_t, uint current_node) {
  BVHNode current = g_bvh_buffer[current_node];
  BVHNode left = g_bvh_buffer[current.left];
  BVHNode right = g_bvh_buffer[current.right];
  float2 l_insp_t = left.bounds.GetInspT(r);
  float2 r_insp_t = right.bounds.GetInspT(r);
  bool left_omittable = IsTraversalOmittable(l_insp_t, current_insp_t);
  bool right_omittable = IsTraversalOmittable(r_insp_t, current_insp_t);
  uint vote = kTraversalOrderNone;
  if(!left_omittable && !right_omittable) {
    if (l_insp_t.x < r_insp_t.x) {
      vote = kTraversalOrderLeftRight;
    } else {
      vote = kTraversalOrderRightLeft;
    }
  } else if (left_omittable) {
    vote = kTraversalOrderOnlyRight;
  } else if (right_omittable) {
    vote = kTraversalOrderOnlyLeft;
  }
  return vote;
}

uint VoteForTraversalOrder(uint t_idx, uint vote) {
  l_traversal_order_votes[t_idx] = vote;
  GroupMemoryBarrierWithGroupSync();
  uint vote_counts[kTraversalOrderMax];
  for (uint i = 0; i < kTraversalOrderMax; i++) {
    vote_counts[i] = 0;
  }
  for (uint vote_idx = 0; vote_idx < kTraverseThreadsPerGroup; vote_idx++) {
    vote_counts[l_traversal_order_votes[vote_idx]] += 1;
  }
  if (vote_counts[kTraversalOrderNone] == kTraverseThreadsPerGroup) {
    return kTraversalOrderNone;
  }
  if (vote_counts[kTraversalOrderNone] + vote_counts[kTraversalOrderOnlyLeft] == kTraverseThreadsPerGroup) {
    return kTraversalOrderOnlyLeft;
  }
  if (vote_counts[kTraversalOrderNone] + vote_counts[kTraversalOrderOnlyRight] == kTraverseThreadsPerGroup) {
    return kTraversalOrderOnlyRight;
  }
  vote_counts[kTraversalOrderLeftRight] += vote_counts[kTraversalOrderOnlyLeft];
  vote_counts[kTraversalOrderRightLeft] += vote_counts[kTraversalOrderOnlyRight];
  if (vote_counts[kTraversalOrderLeftRight] >= vote_counts[kTraversalOrderRightLeft]) {
    return kTraversalOrderLeftRight;
  }
  return kTraversalOrderRightLeft;
}

uint DecideNext(uint left, uint right, uint parent, uint prv, uint order) {
  switch (order) {
    case kTraversalOrderOnlyLeft: {
      return (prv == parent) ? left : parent;
    }
    case kTraversalOrderOnlyRight: {
      return (prv == parent) ? right : parent;
    }
    case kTraversalOrderLeftRight: {
      return (prv == parent) ? left : ((prv == left) ? right : parent);
    }
    case kTraversalOrderRightLeft: {
      return (prv == parent) ? right : ((prv == right) ? left : parent);
    }
  }
  return parent;
}

Interception CastRay(Ray r, uint t_idx) {
  Interception cur_res;
  cur_res.t = -1;
  l_traversal_order_stack[0] = kTraversalOrderUndefined;
  l_traversal_order_stack[t_idx] = 0;
  for (uint cur_vrt = 0, prv_vrt = (uint)-1, nxt_vrt = (uint)-1; cur_vrt != (uint)-1; prv_vrt = cur_vrt, cur_vrt = nxt_vrt) {
    GroupMemoryBarrierWithGroupSync();
    nxt_vrt = g_bvh_buffer[cur_vrt].parent;

    uint current_depth = g_bvh_buffer[cur_vrt].bvh_level;
    if (current_depth == (uint)(-1)) {
      Interception new_insp = CheckLeafPrimitives(r, cur_vrt);
      if (new_insp.t > 0 && (cur_res.t < 0 || new_insp.t < cur_res.t)) {
        cur_res = new_insp;
      }
      continue;
    }
    GroupMemoryBarrierWithGroupSync();
    if (l_traversal_order_stack[current_depth] == kTraversalOrderUndefined) {
      uint vote = CalcTraversalOrderVote(r, cur_res.t, cur_vrt);
      l_traversal_order_stack[current_depth] = VoteForTraversalOrder(t_idx, vote);
    }
    l_traversal_order_stack[current_depth + 1] = kTraversalOrderUndefined;
    uint traversal_order = l_traversal_order_stack[current_depth];
    uint left = g_bvh_buffer[cur_vrt].left;
    uint right = g_bvh_buffer[cur_vrt].right;
    nxt_vrt = DecideNext(left, right, nxt_vrt, prv_vrt, traversal_order);
  }
  return cur_res;
}

#endif // RAYTRACE_TRAVERSE
