#include "utill/transform.h"

#include <glm/gtx/functions.hpp>
#include <glm/gtx/rotate_vector.hpp>

namespace utill {

Transform::Transform(glm::mat4 transform_mat) : tranform_mat_(transform_mat) {}

Transform Transform::Rotation(float rad_x, float rad_y, float rad_z) {
  glm::vec4 dir_x(1, 0, 0, 0);
  dir_x = glm::rotateY(dir_x, rad_y);
  dir_x = glm::rotateZ(dir_x, rad_z);
  glm::vec4 dir_y(0, 1, 0, 0);
  dir_y = glm::rotateX(dir_y, rad_x);
  dir_y = glm::rotateZ(dir_y, rad_z);
  glm::vec4 dir_z(0, 0, 1, 0);
  dir_z = glm::rotateX(dir_z, rad_x);
  dir_z = glm::rotateY(dir_z, rad_y);
  return Transform(glm::mat4(dir_x, dir_y, dir_z, glm::vec4(0, 0, 0, 1)));
}

Transform Transform::Translation(glm::vec3 d_pos) {
  glm::mat4 transform_mat(1.0f);
  transform_mat[3] = glm::vec4(d_pos, 1);
  return Transform(transform_mat);
}

Transform Transform::Combine(const Transform& fst, const Transform& snd) {
  return snd.tranform_mat_ * fst.tranform_mat_;
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