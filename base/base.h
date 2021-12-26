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

extern BaseConfig DEFAULT_BASE_CONFIG;

class Base {
  vk::DynamicLoader dynamic_loader_;
  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT debug_messenger_;

  Base() = default;

  void InitInstance(BaseConfig &config);
  void InitDynamicLoader();
  void InitDebugLogger();

 public:
  static Base& Get();

  void Init(BaseConfig config);

  ~Base();
};

}  // namespace base
