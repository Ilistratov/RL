#ifndef RAYTRACE_COMMON
#define RAYTRACE_COMMON

struct Ray {
  float3 origin;
  float3 direction;
};

struct Interception {
  uint primitive_ind;
  float t;
  float u;
  float v;
};

struct RayTraversalState {
  float4 ray_origin;
  float4 ray_direction;
  Interception intersection;
};

const static uint kTraverseThreadsPerGroup = 32;

struct PerPixelState {
  uint2 pix_cord;
  uint camera_ray_ind;
  uint shadow_ray_ind;
};

#endif // RAYTRACE_COMMON