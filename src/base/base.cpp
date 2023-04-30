#include "base/base.h"

#include <iostream>
#include <vulkan/vulkan_core.h>


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "utill/error_handling.h"
#include "utill/logger.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace base {

void Base::InitInstance(BaseConfig &config) {
  DCHECK(!instance_) << "Instance already initialized";
  bool glfw_init_result = glfwInit();
  CHECK(glfw_init_result) << "failed to init glfw";

  LOG << "Initialized GLFW";

  uint32_t glfw_ext_cnt = 0;
  auto glfw_ext_names_ptr = glfwGetRequiredInstanceExtensions(&glfw_ext_cnt);
  for (uint32_t i = 0; i < glfw_ext_cnt; i++) {
    config.instance_extensions.push_back(glfw_ext_names_ptr[i]);
  }

  vk::ApplicationInfo application_info(config.app_name, VK_API_VERSION_1_3,
                                       config.engine_name);
  vk::InstanceCreateInfo instance_info(
      vk::InstanceCreateFlags{}, &application_info,
      config.instance_layers.size(), config.instance_layers.data(),
      config.instance_extensions.size(), config.instance_extensions.data());

  LOG << "Initializing vk::Instance with\nextensions: "
      << config.instance_extensions << "\nlayers: " << config.instance_layers;

  instance_ = vk::createInstanceUnique(instance_info);
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugMessageFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT /*messageTypes*/,
                 VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
                 void * /*pUserData*/) {
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    LOG << pCallbackData->pMessage;
    return VK_TRUE;
  }
  LOG << pCallbackData->pMessage;
  return VK_FALSE;
}

void Base::InitDebugLogger() {
  LOG << "Initializing Vulkan debug logger";
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

void Base::InitBase(BaseConfig &config) {
  LOG << "Initializing Base";
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  InitInstance(config);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_.get());
  InitDebugLogger();
}

void Base::CreateWindow(vk::Extent2D window_extent) {
  LOG << "Creating window";
  DCHECK(!window_.GetWindow()) << "non-null window during base initialization";
  window_ = Window(window_extent);
}

void Base::CreateContext(ContextConfig &config) {
  LOG << "Creating context";
  DCHECK(!context_.GetDevice()) << "non-null device during base initialization";
  context_ = Context(config);
}

void Base::CreateSwapchain() {
  LOG << "Creating swapchain";
  swapchain_.Create();
}

Base &Base::Get() {
  static Base instance;
  return instance;
}

void Base::Init(BaseConfig config, vk::Extent2D window_extent,
                ContextConfig context_config) {
  InitBase(config);
  CreateWindow(window_extent);
  CreateContext(context_config);
  CreateSwapchain();
}

vk::Instance Base::GetInstance() const { return instance_.get(); }

Window &Base::GetWindow() { return window_; }

Context &Base::GetContext() { return context_; }

Swapchain &Base::GetSwapchain() { return swapchain_; }

Base::~Base() {
  LOG << "Clearing Base";
  swapchain_.Destroy();
  context_ = Context();
  window_ = Window();
  glfwTerminate();
}

} // namespace base
