#pragma once

namespace utill {

struct KeyState {
  int action;
  int mods;
};

struct MouseState {
  KeyState lmb_state;
  KeyState rmb_state;
  double pos_x = 0;
  double pos_y = 0;
  double prv_x = 0;
  double prv_y = 0;
};

class InputManager {
  InputManager();

  static bool IsValidKey(int key);

 public:
  InputManager(const InputManager&) = delete;
  void operator=(const InputManager&) = delete;

  static void Init();
  static KeyState GetKeyState(int key);
  static bool IsKeyPressed(int key);
  static const MouseState& GetMouseState();
};

}  // namespace utill
