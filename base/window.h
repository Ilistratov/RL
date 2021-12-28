#pragma once

#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW\glfw3.h>

namespace base {

class Window {
  GLFWwindow* window_ = nullptr;
  vk::SurfaceKHR surface_;

 public:
  Window() = default;
  Window(vk::Extent2D extent);

  Window(Window&& other) noexcept;
  void operator=(Window&& other) noexcept;
  void Swap(Window& other) noexcept;

  vk::Extent2D GetExtent() const;
  GLFWwindow* GetWindow() const;
  vk::SurfaceKHR GetSurface() const;

  ~Window();
};

}  // namespace base
