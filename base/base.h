#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

namespace base {

struct BaseConfig {
  std::vector<const char *> instance_extensions;
  std::vector<const char *> instance_layers;
  const char* app_name;
  const char* engine_name;
};

/*
 * 'Base' is a singleton class responsible for initializing and managing
 * 'vk::Instance', 'vk::DynamicLoader'm vk::DebugUtilsMessengerEXT.
 * For managing device, queues, swapchain and window 'Context' is used.
 * Context is bound to base in order for it to become accessible globaly.
 * 'Base' is responsible for disposal of 'Context'
 */
class Base {
  vk::DynamicLoader dynamic_loader_;
  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT debug_messenger_;

  Base() = default;

  void InitInstance(BaseConfig &config);
  void InitDebugLogger();

 public:
  static Base& Get();

  void Init(BaseConfig config);

  ~Base();
};

}  // namespace base
