#pragma once

#include <glm/glm.hpp>

namespace utill {

class Transform {
  glm::mat4 tranform_mat_ = glm::mat4(1.0);

  Transform(glm::mat4 transform_mat);

 public:
  Transform() = default;

  static Transform Rotation(float rad_x, float rad_y, float rad_z);
  static Transform Translation(glm::vec3 d_pos);

  static Transform Combine(const Transform& fst, const Transform& snd);

  glm::vec3 GetPos() const;
  glm::vec3 GetDirX() const;
  glm::vec3 GetDirY() const;
  glm::vec3 GetDirZ() const;

  glm::vec3 TransformDir(glm::vec3 dir) const;
  glm::vec3 TransformPoint(glm::vec3 point) const;
};

}  // namespace utill
