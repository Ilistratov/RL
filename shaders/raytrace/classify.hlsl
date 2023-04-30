#ifndef RAYTRACE_CLASSIFY
#define RAYTRACE_CLASSIFY

#include "common.hlsl"

const static float PI = acos(-1);

uint CalcRayHash(Ray r, float3 bb_min, float3 bb_max) {
  int3 origin_cell = ((r.origin - bb_min) / (bb_max - bb_min)) * uint3(128, 128, 128);
  origin_cell = clamp(int3(0, 0, 0), int3(127, 127, 127), origin_cell);
  float vertical_angle = dot(float3(0, 1, 0), r.direction);
  int vertical_angle_cell = (vertical_angle / 2 + 0.5) * 16;
  vertical_angle_cell = clamp(0, 15, vertical_angle_cell);
  float horizontal_angle = atan2(r.origin.x, r.origin.z);
  if (r.origin.y >= 1.0 - 1e-7) {
    horizontal_angle = 0;
  }
  int horizontal_angle_cell = ((horizontal_angle + PI) / (2 * PI)) * 16;
  horizontal_angle_cell = clamp(0, 15, horizontal_angle_cell);
  uint ray_hash = uint(origin_cell.x) | uint(origin_cell.y) << 8 |
                  uint(origin_cell.z) << 16 | uint(horizontal_angle_cell) << 24 |
                  uint(vertical_angle_cell) << 28;
  return ray_hash;
}

#endif //RAYTRACE_CLASSIFY
