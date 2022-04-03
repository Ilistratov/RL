#include "utill/transform.h"

#include <glm/gtx/functions.hpp>
#include <glm/gtx/rotate_vector.hpp>

namespace utill {

Transform::Transform(glm::mat4 transform_mat) : tranform_mat_(transform_mat) {}

Transform Transform::Rotation(float rad, glm::vec3 axis) {
  return glm::rotate(rad, axis);
}

Transform Transform::RotationX(float rad) {
  return Rotation(rad, glm::vec3(1, 0, 0));
}

Transform Transform::RotationY(float rad) {
  return Rotation(rad, glm::vec3(0, 1, 0));
}

Transform Transform::RotationZ(float rad) {
  return Rotation(rad, glm::vec3(0, 0, 1));
}

Transform Transform::Translation(glm::vec3 d_pos) {
  glm::mat4 transform_mat(1.0f);
  transform_mat[3] = glm::vec4(d_pos, 1);
  return Transform(transform_mat);
}

Transform Transform::Combine(const Transform& fst, const Transform& snd) {
  return snd.tranform_mat_ * fst.tranform_mat_;
}

Transform Transform::Combine(const std::vector<Transform>& transforms) {
  Transform res;
  for (auto t : transforms) {
    res.tranform_mat_ = t.tranform_mat_ * res.tranform_mat_;
  }
  return res;
}

glm::vec3 Transform::GetDirX() const {
  return tranform_mat_[0];
}

glm::vec3 Transform::GetDirY() const {
  return tranform_mat_[1];
}

glm::vec3 Transform::GetDirZ() const {
  return tranform_mat_[2];
}

glm::vec3 Transform::GetPos() const {
  return tranform_mat_[3];
}

glm::vec3 Transform::TransformDir(glm::vec3 dir) const {
  return tranform_mat_ * glm::vec4(dir, 0);
}

glm::vec3 Transform::TransformPoint(glm::vec3 point) const {
  return tranform_mat_ * glm::vec4(point, 1);
}

}  // namespace utill