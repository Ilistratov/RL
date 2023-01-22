#include "raytrace/bvh.hlsl"

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

[[vk::binding(8, 0)]] StructuredBuffer<BVHNode> bvh_buffer;

float4 PixCordToCameraSpace(uint pix_x, uint pix_y) {
  float uss_x = (float)(pix_x) / camera_info.screen_width;
  float uss_y = (float)(pix_y) / camera_info.screen_height;
  float2 camera_space_xy = float2(uss_x, uss_y);
  camera_space_xy = camera_space_xy * 2 - float2(1, 1);
  camera_space_xy.x *= camera_info.aspect;
  camera_space_xy.y *= -1;
  return float4(camera_space_xy, 1, 1);
}

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
  float dst;
  uint trg_ind;
  // uint it_count;
  // uint insp_left;
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
  res.a = vertex_pos[vertex_ind[3 * ind + 0].x].xyz;
  res.b = vertex_pos[vertex_ind[3 * ind + 1].x].xyz;
  res.c = vertex_pos[vertex_ind[3 * ind + 2].x].xyz;
  return res;
}

float3 GetNormalAtBarCord(uint trg_ind, float2 bar_cord) {
  float3 na = vertex_normal[vertex_ind[3 * trg_ind + 0].y].xyz;
  float3 nb = vertex_normal[vertex_ind[3 * trg_ind + 1].y].xyz;
  float3 nc = vertex_normal[vertex_ind[3 * trg_ind + 2].y].xyz;
  float3 n = nb * bar_cord.x + nc * bar_cord.y
                             + na * (1 - (bar_cord.x + bar_cord.y));
  return normalize(n);
}

bool IsTraversalOmittable(float2 insp_t, float cur_res) {
  return insp_t.x > insp_t.y || (cur_res > 0 && cur_res + 1e-2 < insp_t.x);
}

Interseption CastRay(Ray r) {
  uint cur_trg = (uint)-1;
  float3 cur_insp = float3(-1, 0, 0);
  // uint insp_l = 0;
  uint n_bvh_nodes = 0;
  uint memb_size = 0;
  // uint it_count = 0;
  bvh_buffer.GetDimensions(n_bvh_nodes, memb_size);

  for (uint cur_vrt = 0, prv_vrt = -1, nxt_vrt = 0; nxt_vrt != -1;
       prv_vrt = cur_vrt, cur_vrt = nxt_vrt) {
    nxt_vrt = bvh_buffer[cur_vrt].parent;
    // ++it_count;

    if (bvh_buffer[cur_vrt].bvh_level == (uint)(-1)) {
      for (uint t_ind = bvh_buffer[cur_vrt].left; t_ind < bvh_buffer[cur_vrt].right; t_ind++) {
        // it_count += 2;
        Triangle t = GetTriangleByInd(t_ind);
        float3 n_insp = t.GetRayInterseption(r);
        if (n_insp.x > 0 && (n_insp.x < cur_insp.x || cur_insp.x < 0)) {
          cur_trg = t_ind;
          cur_insp = n_insp;
          // insp_l = bvh_buffer[cur_vrt].left;
        }
      }
      continue;
    }

    uint fst = bvh_buffer[cur_vrt].left;
    float2 fst_t = bvh_buffer[fst].bounds.GetInspT(r);
    uint snd = bvh_buffer[cur_vrt].right;
    float2 snd_t = bvh_buffer[snd].bounds.GetInspT(r);

    if (snd_t.x < snd_t.y && snd_t.x < fst_t.x) {
      uint tmp = fst;
      fst = snd;
      snd = tmp;
      float2 tmp_t = fst_t;
      fst_t = snd_t;
      snd_t = tmp_t;
    }

    if (prv_vrt == nxt_vrt && !IsTraversalOmittable(fst_t, cur_insp.x)) {
      nxt_vrt = fst;
    } else if (prv_vrt != snd && !IsTraversalOmittable(snd_t, cur_insp.x)) {
      nxt_vrt = snd;
    }
  }

  Interseption res;
  res.trg_ind = cur_trg;
  res.bar_cord = cur_insp.yz;
  // res.it_count = it_count;
  res.dst = cur_insp.x;
  // res.insp_left = insp_l;
  return res;
}

float4 CalcLightAtInterseption(Interseption insp, Ray r) {
  // float t_flag = insp.insp_left / 4;
  // float t_flag2 = insp.insp_left / 16;
  // float t_flag3 = insp.insp_left / 64;
  // float3 materialColor = float3((t_flag % 4) * 0.2 + 0.2, (t_flag2 % 4) * 0.2 + 0.2, (t_flag3 % 4) * 0.2 + 0.2);
  float3 materialColor = float3(1.0, 1.0, 1.0);
	uint specPow = 256;
  Triangle t = GetTriangleByInd(insp.trg_ind);
  float3 insp_point = t.GetPointFromBarCord(insp.bar_cord);
  float3 trg_n = GetNormalAtBarCord(insp.trg_ind, insp.bar_cord);
  //float3 trg_n = -t.GetNormal();

  float diffuse = 0;
  float specular = 0;
  uint light_cnt = 0;
  uint light_struct_size = 0;
  light_pos.GetDimensions(light_cnt, light_struct_size);
  for (int i = 0; i < light_cnt; i++) {
    float3 to_light = light_pos[i].xyz - insp_point;
    float light_dst = length(to_light);
    to_light = normalize(to_light);
    Ray shadow_ray;
    shadow_ray.direction = to_light;
    shadow_ray.origin = insp_point + to_light * 1e-4;
    Interseption shadow_ray_insp = CastRay(shadow_ray);
    bool isValid = shadow_ray_insp.trg_ind == (uint)(-1) | light_dst < shadow_ray_insp.dst;
    diffuse += isValid ? max(0, dot(trg_n, to_light)) : 0;
		specular += isValid ? pow(max(0, dot(to_light, reflect(r.direction, trg_n))), 128) : 0;
  }
  return float4(materialColor * (diffuse + specular + 0.2), 1.0);
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint Gind : SV_GroupIndex) {
  Ray camera_ray = PixCordToRay(DTid.x, DTid.y);
  Interseption res = CastRay(camera_ray);
  color_target[DTid.xy] = float4(0.1, 0.1, 0.1, 1.0);
  depth_target[DTid.xy] = 0.0f;
  if (res.trg_ind != (uint)-1) {
    color_target[DTid.xy] = CalcLightAtInterseption(res, camera_ray);
    depth_target[DTid.xy] = max(0, 1 - res.dst / 100);
  }
  // color_target[DTid.xy] = float4(res.it_count / 128.0, res.it_count / 256.0, res.it_count / 512.0, 1.0);
}
