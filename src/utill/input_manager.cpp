#include "utill/input_manager.h"

#include "base/base.h"

namespace utill {

namespace {

struct InputState {
  KeyState keys[GLFW_KEY_LAST];
  MouseState mouse;
} g_input_state;

void KeyCallback(GLFWwindow* window,
                 int key,
                 int scancode,
                 int action,
                 int mods) {
  if (key == GLFW_KEY_UNKNOWN || key >= GLFW_KEY_LAST) {
    return;
  }
  g_input_state.keys[key] = {action, mods};
}

static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  g_input_state.mouse.pos_x = xpos;
  g_input_state.mouse.pos_y = ypos;
}

}  // namespace

void InputManager::Init() {
  GLFWwindow* window = base::Base::Get().GetWindow().GetWindow();
  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, CursorPosCallback);
}

const KeyState& InputManager::GetKeyState(int key) {
  if (key == GLFW_KEY_UNKNOWN || key >= GLFW_KEY_LAST) {
    return {};
  }
  return g_input_state.keys[key];
}

const MouseState& InputManager::GetMouseState() {
  return g_input_state.mouse;
}

}  // namespace utill
