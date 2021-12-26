#include "base/base.h"

#include <iostream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace base {

BaseConfig DEFAULT_BASE_CONFIG = {
    {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
     VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
    {},
    "RL",
    "RL",
};

Base& Base::Get() {
  static Base instance;
  return instance;
}

static void AddGLFWExtensions(BaseConfig& config) {
  uint32_t glfw_ext_cnt = 0;
  auto glfw_ext_names_ptr = glfwGetRequiredInstanceExtensions(&glfw_ext_cnt);
  for (uint32_t i = 0; i < glfw_ext_cnt; i++) {
    config.instance_extensions.push_back(glfw_ext_names_ptr[i]);
  }
}

void Base::InitInstance(BaseConfig& config) {
  assert(!instance_);
  bool glfw_init_result = glfwInit();
  assert(glfw_init_result);

  vk::ApplicationInfo application_info(config.app_name, VK_API_VERSION_1_2,
                                       config.engine_name);
  AddGLFWExtensions(config);
  vk::InstanceCreateInfo instance_info(
      vk::InstanceCreateFlags{}, &application_info,
      config.instance_layers.size(), config.instance_layers.data(),
      config.instance_extensions.size(), config.instance_layers.data());
  instance_ = vk::createInstanceUnique(instance_info);
  assert(instance_);
}

void Base::InitDynamicLoader() {
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_.get());
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugMessageFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                 VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                 void* /*pUserData*/) {
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cout << "Error: " << pCallbackData->pMessage << '\n';
    return VK_TRUE;
  } else if (messageSeverity >=
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cout << "Warning: " << pCallbackData->pMessage << '\n';
    return VK_FALSE;
  } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cout << "Info: " << pCallbackData->pMessage << '\n';
    return VK_FALSE;
  } else {
    std::cout << "Debug: " << pCallbackData->pMessage << '\n';
    return VK_FALSE;
  }
  //}
}

void Base::InitDebugLogger() {
  using SeverityFlags = vk::DebugUtilsMessageSeverityFlagBitsEXT;
  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      SeverityFlags::eVerbose | SeverityFlags::eInfo | SeverityFlags::eWarning |
      SeverityFlags::eError);
  using TypeFlags = vk::DebugUtilsMessageTypeFlagBitsEXT;
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(TypeFlags::ePerformance |
                                                     TypeFlags::eValidation);
  debug_messenger_ = instance_.get().createDebugUtilsMessengerEXTUnique(
      vk::DebugUtilsMessengerCreateInfoEXT({}, severityFlags, messageTypeFlags,
                                           &debugMessageFunc));
}

void Base::Init(BaseConfig config) {
  InitInstance(config);
  InitDynamicLoader();
  InitDebugLogger();
}

Base::~Base() {
  glfwTerminate();
}

}  // namespace base
