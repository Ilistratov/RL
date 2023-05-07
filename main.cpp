#include <GLFW/glfw3.h>
#include <chrono>
#include <iostream>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

// #include "examples/mandelbrot.h"
#include "examples/raytracer-2.h"
#include "examples/raytracer.h"
#include "examples/test.h"

#include "base/base.h"
#include "base/context.h"
#include "gpu_executor/executor.h"
#include "pipeline_handler/compute.h"
#include "render_data/bvh.h"
#include "render_data/mesh.h"
#include "utill/error_handling.h"
#include "utill/input_manager.h"
#include "utill/logger.h"
#include "utill/transform.h"

const static std::string kSceneObjPath = "../assets/objects/dragon.obj";

void RunAlt() { examples::TestRenderer test; }

void Run() {
  render_data::Mesh mesh = render_data::Mesh::LoadFromObj(kSceneObjPath);
  render_data::BVH bvh =
      render_data::BVH(render_data::BVH::BuildPrimitivesBB(mesh), 24, 12);
  mesh.ReorderPrimitives(bvh.GetPrimitiveOrd());
  examples::RayTracer2 renderer(mesh, bvh);
  auto &window = base::Base::Get().GetWindow();
  utill::Transform initial_pos = utill::Transform::RotationX(0.5) |
                                 utill::Transform::RotationY(-0.7) |
                                 utill::Transform::Translation({10, 15, -10});
  renderer.SetCameraTransform(initial_pos);
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
      {//"VK_LAYER_KHRONOS_validation",
       "VK_LAYER_LUNARG_monitor"},
      //{},
      "RL",
      "RL",
      true,
  };

  base::ContextConfig context_config = {
      {},
      {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
       VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
       VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
       VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
       VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME},
      1,
      vk::QueueFlagBits::eCompute};

  try {
    base::Base::Get().Init(base_config, vk::Extent2D{1280, 768},
                           context_config);
    utill::InputManager::Init();
    Run();
    // RunAlt();
  } catch (std::exception e) {
    LOG << e.what();
  }
  LOG << "RL end";
}
