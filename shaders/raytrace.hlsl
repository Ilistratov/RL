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
  float4 origin;
  float4 direction;
};

Ray PixCordToRay(uint pix_x, uint pix_y) {
  Ray result;
  result.direction = PixCordToCameraSpace(pix_x, pix_y);
  result.direction = mul(camera_info.camera_to_world, result.direction);
  result.origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
  result.direction -= result.origin;
  result.direction[3] = 0;
  result.direction = normalize(result.direction);
  return result;
}

[numthreads(8, 8, 1)] void main(uint3 DTid
                                : SV_DispatchThreadID, uint Gind
                                : SV_GroupIndex) {
  Ray camera_ray = PixCordToRay(DTid.x, DTid.y);
  color_target[DTid.xy] = camera_ray.direction;
  depth_target[DTid.xy] = 0.0f;
}