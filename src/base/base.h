#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "base/context.h"
#include "base/swapchain.h"
#include "base/window.h"

namespace base {

struct BaseConfig {
  std::vector<const char *> instance_extensions;
  std::vector<const char *> instance_layers;
  const char *app_name;
  const char *engine_name;
  bool create_window = true;
};

/*
 * 'Base' is a singleton class responsible for initializing and managing
 * 'vk::Instance', 'vk::DynamicLoader'm vk::DebugUtilsMessengerEXT, 'Window',
 * 'Context' and 'Swapchain'
 */
class Base {
  vk::DynamicLoader dynamic_loader_;
  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT debug_messenger_;

  Window window_;
  Context context_;
  Swapchain swapchain_;

  Base() = default;

  void InitInstance(BaseConfig &config);
  void InitDebugLogger();
  void InitBase(BaseConfig &config);
  void CreateWindow(vk::Extent2D window_extent);
  void CreateContext(ContextConfig &context_config);
  void CreateSwapchain();

public:
  static Base &Get();

  void Init(BaseConfig config, vk::Extent2D window_extent,
            ContextConfig context_config);

  Window &GetWindow();
  Context &GetContext();
  Swapchain &GetSwapchain();

  vk::Instance GetInstance() const;

  ~Base();
};

} // namespace base
