#include "utill/input_manager.h"

#include "base/base.h"

namespace utill {

namespace {

struct InputState {
  KeyState keys[GLFW_KEY_LAST];
  MouseState mouse;
};

static InputState g_input_state;

void KeyCallback(GLFWwindow * /*window*/, int key, int /*scancode*/, int action,
                 int mods) {
  if (key == GLFW_KEY_UNKNOWN || key >= GLFW_KEY_LAST) {
    return;
  }
  g_input_state.keys[key] = {action, mods};
}

static void NormalCursorPosCallback(GLFWwindow * /*window*/, double pos_x,
                                    double pos_y) {
  g_input_state.mouse.pos_x = pos_x;
  g_input_state.mouse.pos_y = pos_y;
}

static void DisabledCursorPosCallback(GLFWwindow *window, double pos_x,
                                      double pos_y) {
  int size_x = 0;
  int size_y = 0;
  glfwGetWindowSize(window, &size_x, &size_y);
  if (pos_y > size_y) {
    pos_y = size_y;
  } else if (pos_y < -size_y) {
    pos_y = -size_y;
  }
  if (pos_x > size_x) {
    double overshoot = pos_x - size_x;
    overshoot -= (int)(overshoot / (2 * size_x)) * (2 * size_x);
    pos_x = -size_x + overshoot;
  } else if (pos_x < -size_x) {
    double overshoot = size_x - pos_x;
    overshoot -= (int)(overshoot / (2 * size_x)) * (2 * size_x);
    pos_x = size_x - overshoot;
  }
  glfwSetCursorPos(window, pos_x, pos_y);
  g_input_state.mouse.pos_x = pos_x / size_x;
  g_input_state.mouse.pos_y = pos_y / size_y;
}

static void MouseButtonCallback(GLFWwindow * /*window*/, int button, int action,
                                int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    g_input_state.mouse.lmb_state = {action, mods};
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    g_input_state.mouse.rmb_state = {action, mods};
  }
}

} // namespace

bool InputManager::IsValidKey(int key) {
  return key != GLFW_KEY_UNKNOWN && key < GLFW_KEY_LAST;
}

void InputManager::Init() {
  GLFWwindow *window = base::Base::Get().GetWindow().GetWindow();
  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, NormalCursorPosCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
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
  return g_input_state.keys[key].action == GLFW_PRESS ||
         g_input_state.keys[key].action == GLFW_REPEAT;
}

MouseState InputManager::GetMouseState() { return g_input_state.mouse; }

void InputManager::SetCursorMode(int mode) {
  GLFWwindow *window = base::Base::Get().GetWindow().GetWindow();
  glfwSetInputMode(window, GLFW_CURSOR, mode);
  if (mode == GLFW_CURSOR_DISABLED) {
    glfwSetCursorPos(window, 0, 0);
    glfwSetCursorPosCallback(window, DisabledCursorPosCallback);
  } else {
    glfwSetCursorPosCallback(window, NormalCursorPosCallback);
  }
}

} // namespace utill
