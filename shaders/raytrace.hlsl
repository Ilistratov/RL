[[vk::binding(0, 0)]] RWTexture2D<float4> color_target;
[[vk::binding(1, 0)]] RWTexture2D<float> depth_target;

[[vk::binding(2, 0)]] StructuredBuffer<float4> vertex_pos;
[[vk::binding(3, 0)]] StructuredBuffer<uint> vertex_ind;
[[vk::binding(4, 0)]] StructuredBuffer<float4> light_pos;

struct CameraInfo {
  float4x4 camera_to_world;
  uint screen_width;
  uint screen_height;
  float aspect;
};

[[vk::binding(5, 0)]] ConstantBuffer<CameraInfo> camera_info;

float4 PixCordToCameraSpace(uint pix_x, uint pix_y) {
  float uss_x = (float)(pix_x) / camera_info.screen_width;
  float uss_y = (float)(pix_y) / camera_info.screen_height;
  float2 camera_space_xy = float2(uss_x, uss_y);
  camera_space_xy = camera_space_xy * 2 - float2(1, 1);
  camera_space_xy.x *= camera_info.aspect;
  return float4(camera_space_xy, 1, 1);
}

struct Ray {
  float3 origin;
  float3 direction;
};

Ray PixCordToRay(uint pix_x, uint pix_y) {
  Ray result;
  float4 dst_pos = PixCordToCameraSpace(pix_x, pix_y);
  dst_pos = mul(camera_info.camera_to_world, dst_pos);
  float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
  result.direction = (dst_pos - origin).xyz;
  result.direction = normalize(result.direction);
  result.origin = origin.xyz;
  return result;
}

float3 GetRayTriangleInterseption(Ray r, float3 a, float3 b, float3 c) {
  float3x3 sys_mat = float3x3(-r.direction, b - a, c - a);
  float det_sys = determinant(sys_mat);
  if (det_sys < 1e-4) {
    return float3(-1, 0, 0);
  }
  float3 sys_coef = r.origin - a;
  float inv_det_sys = 1.0 / det_sys;

  float t = determinant(float3x3(sys_coef, b - a, c - a)) * inv_det_sys;
  if (t < 1e-4) {
    return float3(-1, 0, 0);
  }
  float u = determinant(float3x3(-r.direction, sys_coef, c - a)) * inv_det_sys;
  if (u < 0 || u > 1) {
    return float3(-1, 0, 0);
  }
  float v = determinant(float3x3(-r.direction, b - a, sys_coef)) * inv_det_sys;
  if (v < 0 || v > 1 || u + v > 1) {
    return float3(-1, 0, 0);
  }
  return float3(t, u, v);
}

struct Interseption {
  float3 bar_cord;
  uint trg_ind;
};

Interseption CastRay(Ray r) {
  uint cur_trg = (uint)-1;
  float3 cur_insp = float3(-1, 0, 0);
  uint n_trg = 0;
  uint memb_size = 0;
  vertex_ind.GetDimensions(n_trg, memb_size);

  for (uint i = 0; i < n_trg; i += 3) {
    float3 a = vertex_pos[vertex_ind[i + 0]].xyz;
    float3 b = vertex_pos[vertex_ind[i + 1]].xyz;
    float3 c = vertex_pos[vertex_ind[i + 2]].xyz;
    float3 n_insp = GetRayTriangleInterseption(r, a, b, c);
    if (n_insp.x > 0 && (n_insp.x < cur_insp.x || cur_insp.x < 0)) {
      cur_trg = i / 3;
      cur_insp = n_insp;
    }
  }

  Interseption res;
  res.trg_ind = cur_trg;
  if (cur_trg == (uint)-1) {
    res.bar_cord = cur_insp;
    return res;
  }
  res.bar_cord = float3(cur_insp.y, cur_insp.z, 1 - cur_insp.y - cur_insp.z);
  return res;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint Gind : SV_GroupIndex) {
  Ray camera_ray = PixCordToRay(DTid.x, DTid.y);
  Interseption res = CastRay(camera_ray);
  if (res.trg_ind != (uint)-1) {
    color_target[DTid.xy] = float4(res.bar_cord, 1.0);
  } else {
    float3 nxt_pt = (camera_ray.origin + camera_ray.direction) * 0.25;
    nxt_pt = max(frac(nxt_pt), float3(1, 1, 1) - frac(nxt_pt));
    nxt_pt = pow(nxt_pt, float3(64, 64, 64));
    color_target[DTid.xy] = float4(nxt_pt, 1.0);
  }
  depth_target[DTid.xy] = 0.0f;
}