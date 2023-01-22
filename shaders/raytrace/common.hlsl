#ifndef RAYTRACE_COMMON
#define RAYTRACE_COMMON

struct Ray {
  float3 origin;
  float3 direction;
};

struct Intersection {
  uint primitive_ind;
  float t;
  float u;
  float v;
};

struct RayTraversalState {
  float4 ray_origin;
  float4 ray_direction;
  Intersection intersection;
};

const static uint VECTORIZED_STRUCT_SIZE = 16;

struct VectorizedRayTraversalState {
  float4 ray_origin[VECTORIZED_STRUCT_SIZE];
  float4 ray_direction[VECTORIZED_STRUCT_SIZE];
  Intersection intersection[VECTORIZED_STRUCT_SIZE];

  Ray GetRay(uint tind) {
    Ray r;
    r.origin = ray_origin[tind].xyz;
    r.direction = ray_direction[tind].xyz;
    return r;
  }

  void SetRay(uint tind, Ray r) {
    ray_origin[tind] = float4(r.origin, 1.0);
    ray_direction[tind] = float4(r.direction, 0.0);
  }
};

struct PerPixelState {
  uint2 pix_cord;
  uint camera_ray_ind;
  uint shadow_ray_ind;
};

#endif