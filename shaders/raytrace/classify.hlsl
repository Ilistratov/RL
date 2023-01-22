#include "common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> ray_id_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> ray_class_out;
[[vk::binding(2, 0)]] StructuredBuffer<RayTraversalState> traversal_state_in;

struct ClassificationInfo {
  float4 bb_min;
  float4 bb_max;
};
const static float PI = acos(-1);

[[vk::binding(2, 0)]] ConstantBuffer<ClassificationInfo> classification_info_in;

[numthreads(64, 1, 1)]
void main(uint3 traversal_state_id : SV_DispatchThreadID) {
  uint idx = traversal_state_id.x;
  float4 origin = traversal_state_in[idx].ray_origin;
  int4 origin_cell =
    ((origin - classification_info_in.bb_min) /
    (classification_info_in.bb_max - classification_info_in.bb_min)) *
    uint4(128, 128, 128, 0);
  origin_cell = clamp(int4(0, 0, 0, 0), int4(127, 127, 127, 0), origin_cell);
  float4 direction = traversal_state_in[idx].ray_direction;
  float vertical_angle = dot(float4(0, 1, 0, 0), direction);
  int vertical_angle_cell = (vertical_angle / 2 + 0.5) * 16;
  vertical_angle_cell = clamp(0, 15, vertical_angle_cell);
  float horizontal_angle = atan2(origin.x, origin.z);
  if (origin.y >= 1.0 - 1e-7) {
    horizontal_angle = 0;
  }
  int horizontal_angle_cell = ((horizontal_angle + PI) / (2 * PI)) * 16;
  horizontal_angle_cell = clamp(0, 15, horizontal_angle_cell);
  uint ray_hash = uint(origin_cell.x) | uint(origin_cell.y) << 8 |
                  uint(origin_cell.z) << 16 | uint(horizontal_angle_cell) << 24 |
                  uint(vertical_angle_cell) << 28;
  ray_class_out[idx] = ray_hash;
}
