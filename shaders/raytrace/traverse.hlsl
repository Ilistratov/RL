#ifndef RAYTRACE_TRAVERSE
#define RAYTRACE_TRAVERSE

#include "common.hlsl"
#include "bvh.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<float4> g_vertex_pos;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> g_vertex_ind;
[[vk::binding(2, 0)]] StructuredBuffer<float4> g_vertex_normal;
[[vk::binding(3, 0)]] StructuredBuffer<BVHNode> g_bvh_buffer;

const static uint kQueueSize = 16;
const static uint kThreadsPerGroup = 16;

groupshared VectorizedRayTraversalState l_traversal_state;
groupshared uint l_bvh_node_insp_queue[kQueueSize];
groupshared uint l_queue_end;
groupshared uint l_cur_vrt;
groupshared uint l_prv_vrt;
groupshared uint l_nxt_vrt;

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

void ProcessBvhLeafInspQueue(uint l_ts_ind) {
  Ray r = l_traversal_state.GetRay(l_ts_ind);
  Interception current = l_traversal_state.intersection[l_ts_ind];
  for (uint i = 0; i < l_queue_end; i++) {
    Interception res = CheckLeafPrimitives(r, l_bvh_node_insp_queue[i]);
    if (res.t > 0 && (current.t < 0 || res.t < current.t)) {
      current = res;
    }
  }
  l_traversal_state.intersection[l_ts_ind] = current;
}

void HandleLeaf(uint l_ts_ind) {
  if (l_ts_ind == 0) {
    l_bvh_node_insp_queue[l_queue_end] = l_cur_vrt;
    ++l_queue_end;
  }
  GroupMemoryBarrierWithGroupSync();
  if (l_queue_end < kQueueSize) {
    return;
  }
  ProcessBvhLeafInspQueue(l_ts_ind);
  l_queue_end = 0;
}

bool IsTraversalOmittable(float2 insp_t, float cur_res) {
  return insp_t.x > insp_t.y || (cur_res > 0 && cur_res + 1e-2 < insp_t.x);
}

groupshared int l_traversal_order_votes[kThreadsPerGroup];

uint VoteForTraversalOrder(uint l_ts_ind, uint voted_next) {
  if (voted_next == g_bvh_buffer[l_cur_vrt].left) {
    l_traversal_order_votes[l_ts_ind] = -1;
  } else {
    l_traversal_order_votes[l_ts_ind] = 1;
  }
  GroupMemoryBarrierWithGroupSync();
  int sum = 0;
  for (uint i = 0; i < kThreadsPerGroup; i++) {
    sum += l_traversal_order_votes[i];
  }
  if (sum <= 0) {
    return g_bvh_buffer[l_cur_vrt].left;
  }
  return g_bvh_buffer[l_cur_vrt].right;
}

groupshared bool l_omit_votes[kThreadsPerGroup];

bool VoteForOmitt(uint l_ts_ind, bool vote) {
  l_omit_votes[l_ts_ind] = vote;
  GroupMemoryBarrierWithGroupSync();
  for (uint i = 0; i < kThreadsPerGroup; i++) {
    if (!l_omit_votes[i]) {
      return false;
    }
  }
  return true;
}

void CastRay(uint l_ts_ind, bool any_hit) {
  l_traversal_state.intersection[l_ts_ind].t = -1;
  l_traversal_state.intersection[l_ts_ind].primitive_ind = (uint)-1;
  Ray r = l_traversal_state.GetRay(l_ts_ind);
  l_cur_vrt = 0;
  l_prv_vrt = uint(-1);
  l_queue_end = 0;
  while (l_cur_vrt != (uint)-1) {
    l_nxt_vrt = g_bvh_buffer[l_cur_vrt].parent;
    float2 cur_t = g_bvh_buffer[l_cur_vrt].bounds.GetInspT(r);
    if (VoteForOmitt(l_ts_ind,
            IsTraversalOmittable(cur_t,
                                 l_traversal_state.intersection[l_ts_ind].t))) {
      if (l_ts_ind == 0) {
        l_prv_vrt = l_cur_vrt;
        l_cur_vrt = l_nxt_vrt;
      }
      GroupMemoryBarrierWithGroupSync();
      continue;
    }

    if (g_bvh_buffer[l_cur_vrt].bvh_level == (uint)(-1)) {
      HandleLeaf(l_ts_ind);
      if (any_hit && l_traversal_state.intersection[l_ts_ind].t > 0) {
        return;
      }
      l_prv_vrt = l_cur_vrt;
      l_cur_vrt = l_nxt_vrt;
      continue;
    }

    uint fst = g_bvh_buffer[l_cur_vrt].left;
    float2 fst_t = g_bvh_buffer[fst].bounds.GetInspT(r);
    uint snd = g_bvh_buffer[l_cur_vrt].right;
    float2 snd_t = g_bvh_buffer[snd].bounds.GetInspT(r);

    if (snd_t.x < snd_t.y && snd_t.x < fst_t.x) {
      uint tmp = fst;
      fst = snd;
      snd = tmp;
      float2 tmp_t = fst_t;
      fst_t = snd_t;
      snd_t = tmp_t;
    }

    uint true_fst = VoteForTraversalOrder(l_ts_ind, fst);
    if (true_fst == snd) {
      snd = fst;
      fst = true_fst;
    }

    if (l_ts_ind == 0) {
      if (l_prv_vrt == l_nxt_vrt) {
        l_nxt_vrt = fst;
      } else if (l_prv_vrt == fst) {
        l_nxt_vrt = snd;
      }
      l_prv_vrt = l_cur_vrt;
      l_cur_vrt = l_nxt_vrt;
    }
  }
  ProcessBvhLeafInspQueue(l_ts_ind);
}

#endif // RAYTRACE_TRAVERSE
