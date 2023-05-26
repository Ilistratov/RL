#pragma once

#include <cstdint>
#include <glm/fwd.hpp>
#include <stdint.h>


#include "utill/transform.h"

namespace examples {

struct CameraInfo {
  utill::Transform camera_to_world = {};
  uint32_t screen_width = 0;
  uint32_t screen_height = 0;
  float aspect = 0;
};

class MainCamera {
  CameraInfo camera_info_;
  bool is_controlled_;

public:
  MainCamera() = default;
  MainCamera(uint32_t screen_width, uint32_t screen_height);

  const CameraInfo &GetCameraInfo() const;
  void Update();
  void SetTransform(utill::Transform transform);
};

} // namespace examples
