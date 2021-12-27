#include "base/base.h"

#include <cassert>
#include <iostream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "utill/logger.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace base {

Base& Base::Get() {
  static Base instance;
  return instance;
}

void Base::InitInstance(BaseConfig& config) {
  assert(!instance_);
  bool glfw_init_result = glfwInit();
  assert(glfw_init_result);

  LOG(INFO) << "Initialized GLFW";

  uint32_t glfw_ext_cnt = 0;
  auto glfw_ext_names_ptr = glfwGetRequiredInstanceExtensions(&glfw_ext_cnt);
  for (uint32_t i = 0; i < glfw_ext_cnt; i++) {
    config.instance_extensions.push_back(glfw_ext_names_ptr[i]);
  }

  vk::ApplicationInfo application_info(config.app_name, VK_API_VERSION_1_2,
                                       config.engine_name);
  vk::InstanceCreateInfo instance_info(
      vk::InstanceCreateFlags{}, &application_info,
      config.instance_layers.size(), config.instance_layers.data(),
      config.instance_extensions.size(), config.instance_extensions.data());

  LOG(INFO) << "Initializing vk::Instance with\nextensions: "
            << config.instance_extensions
            << "\nlayers: " << config.instance_layers;

  instance_ = vk::createInstanceUnique(instance_info);
  assert(instance_);
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugMessageFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                 VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                 void* /*pUserData*/) {
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    LOG(ERROR) << pCallbackData->pMessage;
    return VK_TRUE;
  } else if (messageSeverity >=
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    LOG(WARNING) << pCallbackData->pMessage;
    return VK_FALSE;
  } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    LOG(INFO) << pCallbackData->pMessage;
    return VK_FALSE;
  } else {
    LOG(DEBUG) << pCallbackData->pMessage;
    return VK_FALSE;
  }
}

void Base::InitDebugLogger() {
  LOG(INFO) << "Initializing Vulkan debug logger";
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
  LOG(INFO) << "Initializing Base";
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  InitInstance(config);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_.get());
  InitDebugLogger();
  LOG(INFO) << "Initialized Base";
}

Base::~Base() {
  LOG(INFO) << "Clearing Base";
  glfwTerminate();
  LOG(INFO) << "Cleared Base";
}

}  // namespace base
