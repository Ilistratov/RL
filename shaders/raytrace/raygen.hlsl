#include "common.hlsl"

struct CameraInfo {
  float4x4 camera_to_world;
  uint screen_width;
  uint screen_height;
  float aspect;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraInfo> camera_info_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<RayTraversalState> traversal_state_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<PerPixelState> per_pixel_state_out;

float4 PixCordToCameraSpace(uint pix_x, uint pix_y) {
  float uss_x = (float)(pix_x) / camera_info_in.screen_width;
  float uss_y = (float)(pix_y) / camera_info_in.screen_height;
  float2 camera_space_xy = float2(uss_x, uss_y);
  camera_space_xy = camera_space_xy * 2 - float2(1, 1);
  camera_space_xy.x *= camera_info_in.aspect;
  camera_space_xy.y *= -1;
  return float4(camera_space_xy, 1, 1);
}

Ray PixCordToRay(uint pix_x, uint pix_y) {
  Ray result;
  float4 dst_pos = PixCordToCameraSpace(pix_x, pix_y);
  dst_pos = mul(camera_info_in.camera_to_world, dst_pos);
  float4 origin = mul(camera_info_in.camera_to_world, float4(0, 0, 0, 1));
  result.direction = (dst_pos - origin).xyz;
  result.direction = normalize(result.direction);
  result.origin = origin.xyz;
  return result;
}

[numthreads(8, 8, 1)]
void main(uint3 pixel_cord : SV_DispatchThreadID, uint local_index : SV_GroupIndex, uint3 group_index : SV_GroupID) {
  Ray r = PixCordToRay(pixel_cord.x, pixel_cord.y);
  RayTraversalState traversal_state;
  traversal_state.ray_origin = float4(r.origin, 1.0);
  traversal_state.ray_direction = float4(r.direction, 0.0);
  Intersection intersection;
  intersection.primitive_ind = (uint)-1;
  intersection.t = -1.0;
  intersection.u = 0.0;
  intersection.v = 0.0;
  traversal_state.intersection = intersection;

  uint dispatch_width = camera_info_in.screen_width / 8;
  uint dst_offset = (group_index.y * dispatch_width + group_index.x) * 64 + local_index;
  traversal_state_out[dst_offset] = traversal_state;
  per_pixel_state_out[dst_offset].pix_cord = pixel_cord.xy;
  per_pixel_state_out[dst_offset].camera_ray_ind = dst_offset;
  per_pixel_state_out[dst_offset].shadow_ray_ind = (uint)-1;
}
