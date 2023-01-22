#include "common.hlsl"

[[vk::binding(0, 0)]] RWTexture2D<float4> color_target_out;
[[vk::binding(1, 0)]] StructuredBuffer<RayTraversalState> traversal_state_in;
[[vk::binding(2, 0)]] StructuredBuffer<PerPixelState> per_pixel_state_in;

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 group_index : SV_GroupID) {
  PerPixelState ps = per_pixel_state_in[DTid.x];
  RayTraversalState ts = traversal_state_in[ps.camera_ray_ind];
  //color_target_out[ps.pix_cord] = abs(ts.ray_direction);
  uint2 g_ind_2d = uint2(group_index.x % 160, group_index.x / 160);
  color_target_out[ps.pix_cord].xy = g_ind_2d / float2(160, 96);
  color_target_out[ps.pix_cord].w = 1.0;
}
