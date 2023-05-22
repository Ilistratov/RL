#include "common.hlsl"
#include "bvh.hlsl"
#include "classify.hlsl"

//#define VECTORIZED_TRAVERSAL
#define ENABLE_TRAVERSAL_ORDER_OPTIMIZATION
#include "traverse.hlsl"

[[vk::binding(0, 1)]] RWStructuredBuffer<RayTraversalState> g_traversal_state;
[[vk::binding(1, 1)]] RWStructuredBuffer<PerPixelState> g_per_pixel_state;

[[vk::binding(0, 2)]] RWStructuredBuffer<RayTraversalState> g_shadow_ray_traversal_state;
[[vk::binding(1, 2)]] RWStructuredBuffer<uint> g_shadow_ray_hash;

Ray GenerateShadowRay(Interception insp, float3 light_pos) {
  Triangle t = GetTriangleByInd(insp.primitive_ind);
  float2 insp_bar_cord = float2(insp.u, insp.v);
  float3 insp_point = t.GetPointFromBarCord(insp_bar_cord);
  float3 to_light = normalize(light_pos - insp_point);
  Ray shadow_ray;
  shadow_ray.direction = to_light;
  shadow_ray.origin = insp_point + GetNormalAtBarCord(insp.primitive_ind, insp_bar_cord) * 1e-4;
  return shadow_ray;
}

[numthreads(32, 1, 1)]
void main(uint3 global_tidx : SV_DispatchThreadID, uint in_group_tidx : SV_GroupIndex) {
  Ray r;
  r.origin = g_traversal_state[global_tidx.x].ray_origin.xyz;
  r.direction = g_traversal_state[global_tidx.x].ray_direction.xyz;
  Interception insp = CastRay(r, in_group_tidx);
  g_traversal_state[global_tidx.x].intersection = insp;
  uint2 pix_coord = g_per_pixel_state[global_tidx.x].pix_cord;
  if (insp.t > 0) {
    Ray shadow_ray = GenerateShadowRay(insp, float3(20, 30, 20));
    g_shadow_ray_traversal_state[global_tidx.x].ray_direction = float4(shadow_ray.direction, 0.0);
    g_shadow_ray_traversal_state[global_tidx.x].ray_origin = float4(shadow_ray.origin, 1.0);
    BoundingBox bvh_bounds = g_bvh_buffer[0].bounds;
    float3 bb_min = float3(bvh_bounds.x_range.x, bvh_bounds.y_range.x, bvh_bounds.z_range.x);
    float3 bb_max = float3(bvh_bounds.x_range.y, bvh_bounds.y_range.y, bvh_bounds.z_range.y);
    //g_shadow_ray_hash[global_tidx.x] = CalcRayHash(shadow_ray, bb_min, bb_max);
    g_shadow_ray_hash[global_tidx.x] = insp.primitive_ind;
    g_per_pixel_state[global_tidx.x].shadow_ray_ind = global_tidx.x;
  } else {
    g_shadow_ray_hash[global_tidx.x] = (uint)-1;
    g_shadow_ray_traversal_state[global_tidx.x].ray_direction = float4(0, 0, 0, 0);
    g_shadow_ray_traversal_state[global_tidx.x].ray_origin = float4(0, 0, 0, 0);
    g_per_pixel_state[global_tidx.x].shadow_ray_ind = global_tidx.x;
  }
  g_per_pixel_state[global_tidx.x].camera_ray_ind = l_traversal_counters[in_group_tidx];
}
