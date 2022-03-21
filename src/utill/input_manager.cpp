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
  g_input_state.mouse.prv_x = g_input_state.mouse.pos_x;
  g_input_state.mouse.prv_y = g_input_state.mouse.pos_y;
  g_input_state.mouse.pos_x = xpos;
  g_input_state.mouse.pos_y = ypos;
}

static void MouseButtonCallbeck(GLFWwindow* window,
                                int button,
                                int action,
                                int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    g_input_state.mouse.lmb_state = {action, mods};
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    g_input_state.mouse.rmb_state = {action, mods};
  }
}

}  // namespace

bool InputManager::IsValidKey(int key) {
  return key != GLFW_KEY_UNKNOWN && key < GLFW_KEY_LAST;
}

void InputManager::Init() {
  GLFWwindow* window = base::Base::Get().GetWindow().GetWindow();
  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, CursorPosCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallbeck);
}

KeyState InputManager::GetKeyState(int key) {
  if (!IsValidKey(key)) {
    return {};
  }
  return g_input_state.keys[key];
}

bool InputManager::IsKeyPressed(int key) {
  if (!IsValidKey(key)) {
    return false;
  }
  return g_input_state.keys[key].action == GLFW_PRESS;
}

const MouseState& InputManager::GetMouseState() {
  return g_input_state.mouse;
}

}  // namespace utill
