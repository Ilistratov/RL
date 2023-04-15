#include "common.hlsl"
#include "bvh.hlsl"

#define VECTORIZED_TRAVERSAL
#define ENABLE_TRAVERSAL_ORDER_OPTIMIZATION
#include "traverse.hlsl"

[[vk::binding(0, 1)]] RWStructuredBuffer<RayTraversalState> g_traversal_state;
[[vk::binding(1, 1)]] RWStructuredBuffer<PerPixelState> g_per_pixel_state;

[[vk::binding(0, 2)]] RWTexture2D<float4> g_color_target;
[[vk::binding(1, 2)]] RWTexture2D<float> g_depth_target;

float3 GetNormalAtBarCord(uint trg_ind, float2 bar_cord) {
  float3 na = g_vertex_normal[g_vertex_ind[3 * trg_ind + 0].y].xyz;
  float3 nb = g_vertex_normal[g_vertex_ind[3 * trg_ind + 1].y].xyz;
  float3 nc = g_vertex_normal[g_vertex_ind[3 * trg_ind + 2].y].xyz;
  float3 n = nb * bar_cord.x + nc * bar_cord.y
                             + na * (1 - (bar_cord.x + bar_cord.y));
  return normalize(n);
}

float4 CalcLightAtInterseption(Interception insp, Ray r) {
  float3 materialColor = float3(1.0, 1.0, 1.0);
	uint specPow = 256;
  Triangle t = GetTriangleByInd(insp.primitive_ind);
  float2 insp_bar_cord = float2(insp.u, insp.v);
  float3 insp_point = t.GetPointFromBarCord(insp_bar_cord);
  float3 trg_n = GetNormalAtBarCord(insp.primitive_ind, insp_bar_cord);

  float diffuse = 0;
  float specular = 0;
  uint light_cnt = 0;
  uint light_struct_size = 0;
  float3 light_pos = float3(0, 0, 0); //TODO
  float3 to_light = light_pos - insp_point;
  float light_dst = length(to_light);
  to_light = normalize(to_light);
  bool isValid = true; //TODO infinitely more complex staff to handle shadow rays here
  diffuse += isValid ? max(0, dot(trg_n, to_light)) : 0;
	specular += isValid ? pow(max(0, dot(to_light, reflect(r.direction, trg_n))), 128) : 0;
  return float4(materialColor * (diffuse + specular + 0.2), 1.0);
}

[numthreads(32, 1, 1)]
void main(uint3 global_tidx : SV_DispatchThreadID, uint in_group_tidx : SV_GroupIndex) {
  Ray r;
  r.origin = g_traversal_state[global_tidx.x].ray_origin.xyz;
  r.direction = g_traversal_state[global_tidx.x].ray_direction.xyz;
  l_traversal_state.SetRay(in_group_tidx, r);
  uint vrt_visited = CastRay(in_group_tidx, false);
  float visited_fraction = ((float)vrt_visited * 3) / 200;
  Interception insp = l_traversal_state.intersection[in_group_tidx];
  uint2 pix_coord = g_per_pixel_state[global_tidx.x].pix_cord;

  if (insp.t > 0) {
    g_color_target[pix_coord] = CalcLightAtInterseption(l_traversal_state.intersection[in_group_tidx], r);
    g_depth_target[pix_coord] = 1 - exp(-1 / insp.t);
  } else {
    const float4 lower_color = float4(0.2, 0.2, 0.2, 1.0);
    const float4 upper_color = float4(0.6, 0.9, 1.0, 1.0);
    float4 skybox_color = lerp(lower_color, upper_color, (r.direction.y + 1) * 0.5);
    g_color_target[pix_coord] = skybox_color;
    g_depth_target[pix_coord] = 0;
  }
  //g_color_target[pix_coord] = float4(clamp(visited_fraction, 0, 1), clamp(visited_fraction - 1, 0, 1), clamp(visited_fraction - 2, 0, 1), 1);
  //g_color_target[pix_coord] = float4((vrt_visited & 31) / 32.0, ((vrt_visited >> 5) & 31) / 31.0, ((vrt_visited >> 10) & 31) / 31.0, 1);
}
