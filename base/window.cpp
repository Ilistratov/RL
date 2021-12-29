#include "base/window.h"

#include "base/base.h"

namespace base {

Window::Window(vk::Extent2D extent) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window_ = glfwCreateWindow(extent.width, extent.height,
                             "RL2_WND_Name_Configure_TBD", nullptr, nullptr);
  assert(window_);
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  auto res = glfwCreateWindowSurface(Base::Get().GetInstance(), window_,
                                     nullptr, &surface);
  assert(res == VK_SUCCESS);
  surface_ = vk::SurfaceKHR(surface);
}

Window::Window(Window&& other) noexcept {
  *this = std::move(other);
}

void Window::operator=(Window&& other) noexcept {
  Window tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Window::Swap(Window& other) noexcept {
  std::swap(window_, other.window_);
  std::swap(surface_, other.surface_);
}

vk::Extent2D Window::GetExtent() const {
  int width = 0;
  int height = 0;
  glfwGetWindowSize(window_, &width, &height);
  return vk::Extent2D(width, height);
}

GLFWwindow* Window::GetWindow() const {
  return window_;
}

vk::SurfaceKHR Window::GetSurface() const {
  return surface_;
}

Window::~Window() {
  auto instance = base::Base::Get().GetInstance();
  instance.destroySurfaceKHR(surface_);
  glfwDestroyWindow(window_);
}

}  // namespace base
