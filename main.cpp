#include <chrono>
#include <iostream>

#include "examples/raytracer.h"

#include "base/base.h"
#include "base/context.h"
#include "gpu_executer/executer.h"
#include "pipeline_handler/compute.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"

void Run() {
  examples::RayTracer renderer;
  auto& window = base::Base::Get().GetWindow();
  while (!glfwWindowShouldClose(window.GetWindow())) {
    glfwPollEvents();
    if (!renderer.Draw()) {
      LOG << "Failed to draw";
      break;
    }
  }
}

int main() {
  LOG << "RL start";

  base::BaseConfig base_config = {
      {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
       VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
      {"VK_LAYER_KHRONOS_validation"},
      "RL",
      "RL",
  };

  base::ContextConfig context_config = {
      {},
      {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
       VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
       VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME},
      2,
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics};

  try {
    base::Base::Get().Init(base_config, vk::Extent2D{1366, 768},
                           context_config);
    utill::InputManager::Init();
    Run();
  } catch (std::exception e) {
    LOG << e.what();
  }
  LOG << "RL end";
}
