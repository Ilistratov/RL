#include <chrono>
#include <iostream>

#include "base/base.h"
#include "base/context.h"
#include "descriptor_handler/pool.h"
#include "gpu_executer/executer.h"
#include "gpu_resources/image_manager.h"
#include "pipeline_handler/compute.h"
#include "utill/logger.h"

/*
using descriptor_handler::Pool;
using gpu_resources::DeviceMemoryAllocator;
using gpu_resources::ImageManager;
using pipeline_handler::Compute;

class RenderTargetImageAdapter : public descriptor_handler::Binding {
  gpu_resources::Image* image_ = nullptr;
  vk::ImageView image_view_;

 public:
  RenderTargetImageAdapter(const RenderTargetImageAdapter&) = delete;
  void operator=(const RenderTargetImageAdapter&) = delete;

  RenderTargetImageAdapter() = default;
  RenderTargetImageAdapter(uint32_t user_ind, ImageManager& image_manager) {
    gpu_resources::ImageUsage usage;
    usage.access = vk::AccessFlagBits2KHR::eShaderWrite |
                   vk::AccessFlagBits2KHR::eShaderRead;
    usage.layout = vk::ImageLayout::eGeneral;
    usage.stage = vk::PipelineStageFlagBits2KHR::eComputeShader;
    usage.usage = vk::ImageUsageFlagBits::eStorage;
    image_manager.AddUsage(user_ind, usage);
    image_ = image_manager.GetImage();
  }

  void Swap(RenderTargetImageAdapter& other) noexcept {
    std::swap(image_, other.image_);
    std::swap(image_view_, other.image_view_);
  }

  void CreateImageView() {
    assert(!image_view_);
    auto device = base::Base::Get().GetContext().GetDevice();
    image_view_ = device.createImageView(vk::ImageViewCreateInfo(
        {}, image_->GetImage(), vk::ImageViewType::e2D, image_->GetFormat(), {},
        image_->GetSubresourceRange()));
  }

  vk::DescriptorSetLayoutBinding GetVkBinding() const noexcept {
    return vk::DescriptorSetLayoutBinding({}, vk::DescriptorType::eStorageImage,
                                          1, vk::ShaderStageFlagBits::eCompute,
                                          {});
  }

  descriptor_handler::Write GetWrite() const noexcept {
    descriptor_handler::Write res;
    vk::DescriptorImageInfo image_info({}, image_view_,
                                       vk::ImageLayout::eGeneral);
    res.type = vk::DescriptorType::eStorageImage;
    res.image_info = {image_info};
    return res;
  }

  ~RenderTargetImageAdapter() {
    auto device = base::Base::Get().GetContext().GetDevice();
    device.destroyImageView(image_view_);
  }
};

struct PushConstants {
  uint32_t s_width = 0;
  uint32_t s_height = 0;
  float center_x = 0;
  float center_y = 0;
  float scale = 1.0;
};

struct PerFrame {
  Pool descriptor_pool;
  Compute pipeline;
  DeviceMemoryAllocator memory_allocator;
  ImageManager render_target;
  ImageManager swapchain_image;
  RenderTargetImageAdapter render_target_adapter;
  gpu_executer::Executer executer;
  vk::Semaphore transfer_finished_semaphore;
  PushConstants pc;

  void DeclareImageUsage() {
    gpu_resources::ImageUsage rt_init;
    rt_init.layout = vk::ImageLayout::eUndefined;
    rt_init.stage = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
    render_target.AddUsage(0, rt_init);

    RenderTargetImageAdapter rt_adapter(1, render_target);
    render_target_adapter.Swap(rt_adapter);

    gpu_resources::ImageUsage rt_transfer;
    rt_transfer.layout = vk::ImageLayout::eTransferSrcOptimal;
    rt_transfer.stage = vk::PipelineStageFlagBits2KHR::eTransfer;
    rt_transfer.access = vk::AccessFlagBits2KHR::eTransferRead;
    rt_transfer.usage = vk::ImageUsageFlagBits::eTransferSrc;
    render_target.AddUsage(2, rt_transfer);

    gpu_resources::ImageUsage swp_tranfer;
    swp_tranfer.layout = vk::ImageLayout::eTransferDstOptimal;
    swp_tranfer.access = vk::AccessFlagBits2KHR::eTransferWrite;
    swp_tranfer.stage = vk::PipelineStageFlagBits2KHR::eTransfer;
    swp_tranfer.usage = vk::ImageUsageFlagBits::eTransferDst;

    gpu_resources::ImageUsage swp_present;
    swp_present.layout = vk::ImageLayout::ePresentSrcKHR;
    swp_present.stage = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
  }

  void CreateResourcesAndAllocateMemory() {
    render_target.CreateImage();
    render_target.ReserveMemoryBlock(memory_allocator);
    memory_allocator.Allocate();
    auto bind_info = render_target.GetBindMemoryInfo(memory_allocator);
    auto device = base::Base::Get().GetContext().GetDevice();
    device.bindImageMemory2(bind_info);
  }

  void CreatePipeline() {
    vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0,
                                   sizeof(PushConstants));
    pipeline = Compute({&render_target_adapter}, descriptor_pool, {pc_range},
                       "plain_color.spv", "main");
    descriptor_pool.Create();
    render_target_adapter.CreateImageView();
    pipeline.UpdateDescriptorSet({&render_target_adapter});
  }

  void CreateTasks() {
    auto no_record_lambda = [](vk::CommandBuffer cmd) {};
    auto* init_rt_task_ptr = new gpu_executer::LambdaTask(
        vk::PipelineStageFlagBits2KHR::eTopOfPipe, no_record_lambda, {}, {});
    auto init_rt_task = std::unique_ptr<gpu_executer::Task>(init_rt_task_ptr);

    auto dispatch_lambda = [this](vk::CommandBuffer cmd) {
      auto& swapchain = base::Base::Get().GetSwapchain();
      cmd.pushConstants(pipeline.GetLayout(), vk::ShaderStageFlagBits::eCompute,
                        0u, sizeof(PushConstants), &pc);
      pipeline.RecordDispatch(cmd, swapchain.GetExtent().width / 8,
                              swapchain.GetExtent().height / 8, 1);
    };
    auto* dispatch_task_ptr = new gpu_executer::LambdaTask(
        vk::PipelineStageFlagBits2KHR::eComputeShader, dispatch_lambda, {}, {});
    auto dispatch_task = std::unique_ptr<gpu_executer::Task>(dispatch_task_ptr);

    auto transfer_lambda = [this](vk::CommandBuffer cmd) {
      auto pre_blit_barrier = swapchain_image.GetImage()->GetBarrier(
          vk::PipelineStageFlagBits2KHR::eTopOfPipe, {},
          vk::PipelineStageFlagBits2KHR::eTransfer,
          vk::AccessFlagBits2KHR::eTransferWrite, vk::ImageLayout::eUndefined,
          vk::ImageLayout::eTransferDstOptimal);
      cmd.pipelineBarrier2KHR(
          vk::DependencyInfoKHR({}, {}, {}, pre_blit_barrier));

      gpu_resources::Image::RecordBlit(cmd, *render_target.GetImage(),
                                       *swapchain_image.GetImage());

      auto post_blit_barrier = swapchain_image.GetImage()->GetBarrier(
          vk::PipelineStageFlagBits2KHR::eTransfer,
          vk::AccessFlagBits2KHR::eTransferWrite,
          vk::PipelineStageFlagBits2KHR::eBottomOfPipe, {},
          vk::ImageLayout::eTransferDstOptimal,
          vk::ImageLayout::ePresentSrcKHR);
      cmd.pipelineBarrier2KHR(
          vk::DependencyInfoKHR({}, {}, {}, post_blit_barrier));
    };
    auto device = base::Base::Get().GetContext().GetDevice();
    transfer_finished_semaphore = device.createSemaphore({});
    auto& swapchain = base::Base::Get().GetSwapchain();
    auto* transfer_task_ptr = new gpu_executer::LambdaTask(
        vk::PipelineStageFlagBits2KHR::eTransfer, transfer_lambda,
        swapchain.GetImageAvaliableSemaphore(), transfer_finished_semaphore);
    auto transfer_task = std::unique_ptr<gpu_executer::Task>(transfer_task_ptr);

    executer.ScheduleTask(std::move(init_rt_task));
    executer.ScheduleTask(std::move(dispatch_task));
    executer.ScheduleTask(std::move(transfer_task));

    auto dep_map = render_target.GetBarriers();
    for (const auto& [task_ind, barrier] : dep_map) {
      executer.AddBarrier(task_ind, barrier);
    }
  }

  void Init(uint32_t swapchain_image_ind) {
    render_target = ImageManager::CreateStorageImage();
    swapchain_image = ImageManager::CreateSwapchainImage(swapchain_image_ind);
    DeclareImageUsage();
    CreateResourcesAndAllocateMemory();
    CreatePipeline();
    CreateTasks();
    executer.PreExecuteInit();
  }

  ~PerFrame() {
    auto device = base::Base::Get().GetContext().GetDevice();
    device.waitIdle();
    device.destroySemaphore(transfer_finished_semaphore);
  }
};

double g_CAMERA_POS_X = 0.0;
double g_CAMERA_POS_Y = 0.0;
double g_SCALE = 1.0;
double g_TARGET_SCALE = 1.0;
double g_SENSiTIVITY = 0.01;
const int kSmoothingRate = 8;

void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
  g_CAMERA_POS_X += xpos * g_SCALE * g_SENSiTIVITY;
  g_CAMERA_POS_Y += ypos * g_SCALE * g_SENSiTIVITY;
  glfwSetCursorPos(window, 0.0, 0.0);
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
    g_TARGET_SCALE *= 0.5;
  } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    g_TARGET_SCALE *= 2;
  }
}

void WindowFocusCallback(GLFWwindow* window, int focused) {
  if (focused) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(window, 0.0, 0.0);
    glfwSetCursorPosCallback(window, CursorPositionCallback);
    glfwSetMouseButtonCallback(window, MouseButtonCallback);
  } else {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window, nullptr);
    glfwSetMouseButtonCallback(window, nullptr);
  }
}

void InitInput() {
  auto window = base::Base::Get().GetWindow().GetWindow();
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPos(window, 0.0, 0.0);
  glfwSetCursorPosCallback(window, CursorPositionCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  // glfwSetWindowFocusCallback(window, WindowFocusCallback);
}

void Run() {
  auto& swapchain = base::Base::Get().GetSwapchain();
  std::vector<PerFrame> prt(swapchain.GetImageCount());
  for (uint32_t i = 0; i < prt.size(); i++) {
    prt[i].Init(i);
  }
  InitInput();
  auto& window = base::Base::Get().GetWindow();
  LOG(INFO) << "Ready to render";
  while (!glfwWindowShouldClose(window.GetWindow())) {
    glfwPollEvents();
    if (!swapchain.AcquireNextImage()) {
      LOG(ERROR) << "Failed to acquire next image";
      break;
    }
    uint64_t ind = swapchain.GetActiveImageInd();
    prt[ind].pc.s_width = swapchain.GetExtent().width;
    prt[ind].pc.s_height = swapchain.GetExtent().height;
    prt[ind].pc.center_x = g_CAMERA_POS_X;
    prt[ind].pc.center_y = g_CAMERA_POS_Y;
    g_SCALE =
        (g_SCALE * kSmoothingRate + g_TARGET_SCALE) / (kSmoothingRate + 1);
    prt[ind].pc.scale = g_SCALE;
    prt[ind].executer.Execute();
    if (swapchain.Present(prt[ind].transfer_finished_semaphore) !=
        vk::Result::eSuccess) {
      LOG(ERROR) << "Failed to present";
      break;
    }
  }
}
*/

/* TODO apparently it is invalid to do layout transition with dst layout
 * being eUndefined. Image manager needs to be fixed. Temporal fix for now:
 * in ImageManager::GetBarriers, if dst_layout of barrier if Undefined or
 * Preinitialized, then change it to match src_layout
 */

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
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics};

  try {
    base::Base::Get().Init(base_config, vk::Extent2D{1280, 768},
                           context_config);
    // Run();
  } catch (std::exception e) {
    LOG(ERROR) << e.what();
  }

  LOG(INFO) << "RL end";
}
