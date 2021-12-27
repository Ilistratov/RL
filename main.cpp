#include <iostream>
#include <source_location>

#include "base/base.h"
#include "utill/logger.h"

int main() {
  LOG(INFO) << "RL start";
  base::BaseConfig config = {
    {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
     VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
    {"VK_LAYER_KHRONOS_validation"},
    "RL", "RL",};
  try {
    base::Base::Get().Init(config);
  } catch(std::exception e) {
    LOG(ERROR) << e.what();
  }
  LOG(INFO) << "RL end";
}
