#include "common.hlsl"
#include "traverse.hlsl"

[[vk::binding(0, 1)]] RWStructuredBuffer<RayTraversalState> g_traversal_state;
[[vk::binding(1, 1)]] RWStructuredBuffer<PerPixelState> g_per_pixel_state;

[[vk::binding(0, 2)]] RWTexture2D<float4> g_color_target;
[[vk::binding(1, 2)]] RWTexture2D<float> g_depth_target;

[[vk::binding(0, 3)]] RWStructuredBuffer<RayTraversalState> g_shadow_ray_traversal_state;
[[vk::binding(1, 3)]] RWStructuredBuffer<uint> g_shadow_ray_ord;

float4 CalcLightAtInterseption(Interception primary_insp, Ray primary_ray, Interception shadow_insp, Ray shadow_ray) {
  float3 materialColor = float3(1.0, 1.0, 1.0);
	uint specPow = 256;
  Triangle t = GetTriangleByInd(primary_insp.primitive_ind);
  float2 insp_bar_cord = float2(primary_insp.u, primary_insp.v);
  float3 trg_n = GetNormalAtBarCord(primary_insp.primitive_ind, insp_bar_cord);

  float diffuse = 0;
  float specular = 0;
  uint light_cnt = 0;
  uint light_struct_size = 0;
  float3 ligt_pos = float3(0, 0, 0);//TODO
  float3 to_light = shadow_ray.direction;
  float dst_to_light = distance(shadow_ray.origin, ligt_pos);
  bool is_lightsource_visible = shadow_insp.t < 0 || shadow_insp.t > dst_to_light;
  diffuse += is_lightsource_visible ? max(0, dot(trg_n, to_light)) : 0;
	specular += is_lightsource_visible ? pow(max(0, dot(to_light, reflect(primary_ray.direction, trg_n))), 128) : 0;
  return float4(materialColor * (diffuse + specular + 0.2), 1.0);
}

float4 CounterToHeat(uint counter) {
  float val = counter / 100.0;
  return float4(clamp(val, 0, 1), clamp(val - 1, 0, 1), clamp(val - 2, 0, 1), 1.0);
}

[numthreads(32, 1, 1)]
void main(uint3 global_tidx : SV_DispatchThreadID, uint in_group_tidx : SV_GroupIndex) {
  uint pix_state_idx = g_shadow_ray_ord[global_tidx.x];
  //uint pix_state_idx = global_tidx.x;
  uint2 pix_coord = g_per_pixel_state[pix_state_idx].pix_cord;
  Ray primary_ray;
  primary_ray.origin = g_traversal_state[pix_state_idx].ray_origin.xyz;
  primary_ray.direction = g_traversal_state[pix_state_idx].ray_direction.xyz;
  Interception primary_insp = g_traversal_state[pix_state_idx].intersection;
  l_traversal_counters[in_group_tidx] = 0;
  if (primary_insp.t > 0) {
    Ray shadow_ray;
    shadow_ray.origin = g_shadow_ray_traversal_state[pix_state_idx].ray_origin.xyz;
    shadow_ray.direction = g_shadow_ray_traversal_state[pix_state_idx].ray_direction.xyz;
    Interception shadow_insp;
    shadow_insp = CastRay(shadow_ray, in_group_tidx);
    g_color_target[pix_coord] = CalcLightAtInterseption(primary_insp, primary_ray, shadow_insp, shadow_ray);
    g_depth_target[pix_coord] = 1 - exp(-1 / primary_insp.t);
  } else {
    const float4 lower_color = float4(0.2, 0.2, 0.2, 1.0);
    const float4 upper_color = float4(0.6, 0.9, 1.0, 1.0);
    float4 skybox_color = lerp(lower_color, upper_color, (primary_ray.direction.y + 1) * 0.5);
    g_color_target[pix_coord] = skybox_color;
    g_depth_target[pix_coord] = 0;
  }
  //g_color_target[pix_coord] = CounterToHeat(l_traversal_counters[in_group_tidx]);
  //g_color_target[pix_coord] = CounterToHeat(g_per_pixel_state[pix_state_idx].camera_ray_ind);
  if (pix_coord.x % 4 == 0 && pix_coord.y % 4 == 0) {
    g_color_target[pix_coord] = CounterToHeat(g_per_pixel_state[pix_state_idx].camera_ray_ind + l_traversal_counters[in_group_tidx]);
  }
  if (pix_coord.x < 10 && pix_coord.y <= 300) {
    g_color_target[pix_coord] = CounterToHeat(300 - pix_coord.y);
  } else if (pix_coord.x < 15 && pix_coord.y <= 305) {
    g_color_target[pix_coord] = float4(0, 0, 0, 1.0);
  }
}