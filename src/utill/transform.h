#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace utill {

class Transform {
  glm::mat4 tranform_mat_ = glm::mat4(1.0);

  Transform(glm::mat4 transform_mat);

public:
  Transform() = default;

  static Transform Rotation(float rad, glm::vec3 axis);
  static Transform RotationX(float rad);
  static Transform RotationY(float rad);
  static Transform RotationZ(float rad);
  static Transform Rotation(float rad_x, float rad_y, float rad_z,
                            glm::vec3 origin);
  static Transform Translation(glm::vec3 d_pos);

  static Transform Combine(const Transform &fst, const Transform &snd);
  static Transform Combine(const std::vector<Transform> &transforms);

  glm::vec3 GetPos() const;
  glm::vec3 GetDirX() const;
  glm::vec3 GetDirY() const;
  glm::vec3 GetDirZ() const;

  glm::vec3 TransformDir(glm::vec3 dir) const;
  glm::vec3 TransformPoint(glm::vec3 point) const;
};

Transform operator|(const Transform &fst, const Transform &snd);

} // namespace utill
