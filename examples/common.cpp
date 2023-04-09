#include "common.h"
#include "base/base.h"
#include "utill/input_manager.h"
#include "utill/transform.h"

namespace examples {

const static float PI = acos(-1);

MainCamera::MainCamera(uint32_t screen_width, uint32_t screen_height)
    : camera_info_{utill::Transform(), screen_width, screen_height,
                   float(screen_width) / screen_height},
      is_controlled_(false) {}

const CameraInfo& MainCamera::GetCameraInfo() const {
  return camera_info_;
}

const static int g_move_axis[][2] = {{GLFW_KEY_A, GLFW_KEY_D},
                                     {GLFW_KEY_LEFT_SHIFT, GLFW_KEY_SPACE},
                                     {GLFW_KEY_S, GLFW_KEY_W}};

static float GetAxisVal(int axis) {
  if (utill::InputManager::IsKeyPressed(g_move_axis[axis][0])) {
    return -1;
  } else if (utill::InputManager::IsKeyPressed(g_move_axis[axis][1])) {
    return 1;
  }
  return 0;
}

void MainCamera::Update() {
  auto m_state = utill::InputManager::GetMouseState();
  if (m_state.lmb_state.action == GLFW_PRESS && !is_controlled_) {
    is_controlled_ = true;
    utill::InputManager::SetCursorMode(GLFW_CURSOR_DISABLED);
    m_state = utill::InputManager::GetMouseState();
  } else if (utill::InputManager::IsKeyPressed(GLFW_KEY_ESCAPE) &&
             is_controlled_) {
    utill::InputManager::SetCursorMode(GLFW_CURSOR_NORMAL);
    is_controlled_ = false;
  }
  if (!is_controlled_) {
    return;
  }

  float c_ang_x = m_state.pos_y * (PI / 2);
  float c_ang_y = m_state.pos_x * PI;

  utill::Transform rotate_y = utill::Transform::RotationY(c_ang_y);
  utill::Transform rotate_x =
      utill::Transform::Rotation(c_ang_x, rotate_y.GetDirX());
  auto pos = camera_info_.camera_to_world.GetPos();
  camera_info_.camera_to_world = utill::Transform::Combine(rotate_y, rotate_x);
  glm::vec3 move_dir = camera_info_.camera_to_world.GetDirX() * GetAxisVal(0) +
                       camera_info_.camera_to_world.GetDirY() * GetAxisVal(1) +
                       camera_info_.camera_to_world.GetDirZ() * GetAxisVal(2);
  if (glm::length(move_dir) >= 1) {
    move_dir = glm::normalize(move_dir);
  }
  utill::Transform translate =
      utill::Transform::Translation(pos + move_dir * 1.0f);
  camera_info_.camera_to_world =
      utill::Transform::Combine(camera_info_.camera_to_world, translate);
}

void MainCamera::SetPos(glm::vec3 pos) {
  utill::Transform& current = camera_info_.camera_to_world;
  utill::Transform translate =
      utill::Transform::Translation(pos - current.GetPos());
  current = utill::Transform::Combine(current, translate);
}

}  // namespace examples
