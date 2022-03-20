#pragma once

namespace utill {

struct KeyState {
  int action;
  int mods;
};

struct MouseState {
  KeyState lmb_state;
  KeyState rmb_state;
  double pos_x;
  double pos_y;
};

class InputManager {
  InputManager();

 public:
  InputManager(const InputManager&) = delete;
  void operator=(const InputManager&) = delete;

  static void Init();
  static const KeyState& GetKeyState(int key);
  static const MouseState& GetMouseState();
};

}  // namespace utill
