#include <iostream>

#include "base/base.h"
#include "base/context.h"
#include "descriptor_handler/pool.h"
#include "gpu_executer/executer.h"
#include "gpu_resources/image_manager.h"
#include "pipeline_handler/compute.h"
#include "utill/logger.h"

struct PerFrame {
  descriptor_handler::Pool descriptor_pool;
  pipeline_handler::Compute pipeline;
};

int main() {
  LOG(INFO) << "RL start";

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
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics |
          vk::QueueFlagBits::eTransfer};

  try {
    base::Base::Get().Init(base_config, vk::Extent2D{1280, 768},
                           context_config);
  } catch (std::exception e) {
    LOG(ERROR) << e.what();
  }

  LOG(INFO) << "RL end";
}
