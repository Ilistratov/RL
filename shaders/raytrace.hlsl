[[vk::binding(0, 0)]] RWTexture2D<float4> color_target;
[[vk::binding(1, 0)]] RWTexture2D<float> depth_target;

[[vk::binding(2, 0)]] StructuredBuffer<float4> vertex_pos;
[[vk::binding(3, 0)]] StructuredBuffer<float4> vertex_normal;
[[vk::binding(4, 0)]] StructuredBuffer<float2> vertex_texcoord;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> vertex_ind;
[[vk::binding(6, 0)]] StructuredBuffer<float4> light_pos;

struct CameraInfo {
  float4x4 camera_to_world;
  uint screen_width;
  uint screen_height;
  float aspect;
};

[[vk::binding(7, 0)]] ConstantBuffer<CameraInfo> camera_info;

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

struct Interseption {
  float2 bar_cord;
  uint trg_ind;
};

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
};

Triangle GetTriangleByInd(uint ind) {
  Triangle res;
  res.a = vertex_pos[vertex_ind[3 * ind + 0].x].xyz;
  res.b = vertex_pos[vertex_ind[3 * ind + 1].x].xyz;
  res.c = vertex_pos[vertex_ind[3 * ind + 2].x].xyz;
  return res;
}

Interseption CastRay(Ray r) {
  uint cur_trg = (uint)-1;
  float3 cur_insp = float3(-1, 0, 0);
  uint n_trg = 0;
  uint memb_size = 0;
  vertex_ind.GetDimensions(n_trg, memb_size);

  for (uint i = 0; i * 3 < n_trg; ++i) {
    Triangle t = GetTriangleByInd(i);
    float3 n_insp = t.GetRayInterseption(r);
    if (n_insp.x > 0 && (n_insp.x < cur_insp.x || cur_insp.x < 0)) {
      cur_trg = i;
      cur_insp = n_insp;
    }
  }

  Interseption res;
  res.trg_ind = cur_trg;
  res.bar_cord = cur_insp.yz;
  return res;
}

float4 CalcLightAtInterseption(Interseption insp, Ray r) {
  float3 materialColor = float3(1, 1, 1);
	uint specPow = 256;
  Triangle t = GetTriangleByInd(insp.trg_ind);
  float3 insp_point = t.GetPointFromBarCord(insp.bar_cord);
  float3 trg_n = t.GetNormal();

  float diffuse = 0;
  float specular = 0;
  uint light_cnt = 0;
  uint light_struct_size = 0;
  light_pos.GetDimensions(light_cnt, light_struct_size);
  for (int i = 0; i < light_cnt; i++) {
    float3 to_light = light_pos[i].xyz - insp_point;
    float3 light_dst = length(to_light);
    to_light = normalize(to_light);
    Ray shadow_ray;
    shadow_ray.direction = to_light;
    shadow_ray.origin = insp_point + to_light * 1e-3;
    Interseption shadow_ray_insp = CastRay(shadow_ray);
    if (shadow_ray_insp.trg_ind != (uint)(-1)) {
      continue;
    }
    diffuse += max(0, dot(trg_n, to_light));
		specular += pow(max(0, dot(to_light, -reflect(r.direction, trg_n))), 128);
  }
  return float4(materialColor * (diffuse + specular), 1.0);
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint Gind : SV_GroupIndex) {
  Ray camera_ray = PixCordToRay(DTid.x, DTid.y);
  Interseption res = CastRay(camera_ray);
  if (res.trg_ind != (uint)-1) {
    color_target[DTid.xy] = CalcLightAtInterseption(res, camera_ray);
  } else {
    float3 nxt_pt = (camera_ray.origin + camera_ray.direction) * 0.25;
    nxt_pt = max(frac(nxt_pt), float3(1, 1, 1) - frac(nxt_pt));
    nxt_pt = pow(nxt_pt, float3(64, 64, 64));
    nxt_pt = max(nxt_pt, float3(0.1, 0.1, 0.1));
    color_target[DTid.xy] = float4(nxt_pt, 1.0);
    depth_target[DTid.xy] = 0.0f;
  }
}
