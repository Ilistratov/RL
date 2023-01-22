#ifndef RAYTRACE_BVH
#define RAYTRACE_BVH

#include "common.hlsl"

const static float2 T_NEVER_INTERSECT = float2(0, -1);
const static float2 T_ALWAYS_INTERSECT = float2(0, 1e9);

float2 GetInspT1d(float origin, float speed, float2 range) {
  int is_inside = range.x < origin && origin < range.y ? 1 : 0;
  int speed_is_zero = abs(speed) < 1e-7 ? 1 : 0;
  float2 t = (range - float2(origin, origin)) / float2(speed, speed);
  float tmx = max(t.x, t.y);
  float tmn = min(t.x, t.y);
  tmn *= (1 - is_inside);
  
  return (1 - speed_is_zero) * float2(tmn, tmx) +
         speed_is_zero * (is_inside * T_ALWAYS_INTERSECT + (1 - is_inside) * T_NEVER_INTERSECT);
}

struct BoundingBox {
  float2 x_range;
  float2 y_range;
  float2 z_range;

  float2 GetInspT(Ray r) {
    float2 t_range_x = GetInspT1d(r.origin.x, r.direction.x, x_range);
    float2 t_range_y = GetInspT1d(r.origin.y, r.direction.y, y_range);
    float2 t_range_z = GetInspT1d(r.origin.z, r.direction.z, z_range);
    float2 res;
    res.x = max(max(t_range_x.x, t_range_y.x), t_range_z.x);
    res.y = min(min(t_range_x.y, t_range_y.y), t_range_z.y);
    return res;
  }
};

struct BVHNode {
  BoundingBox bounds;
  uint left;
  uint right;
  uint parent;
  uint bvh_level;
};

#endif
