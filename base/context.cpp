#include "base/context.h"

#include <map>

#include "base/base.h"
#include "utill/logger.h"

namespace base {

namespace {

class PhysicalDevicePicker {
  const ContextConfig* config_;
  vk::SurfaceKHR surface_;
  std::map<std::string, char> extension_availability_;
  vk::PhysicalDevice result_device_;
  uint32_t result_queue_family_index_ = -1;

  bool CheckFeatures(vk::PhysicalDevice device) const {
    return device.getFeatures().geometryShader;
  }

  bool CheckPresentModes(vk::PhysicalDevice device) const {
    return !device.getSurfacePresentModesKHR(surface_).empty() &&
           !device.getSurfaceFormatsKHR(surface_).empty();
  }

  uint32_t GetSuitableQueueFamilyIndex(vk::PhysicalDevice device) const {
    auto queue_properties = device.getQueueFamilyProperties();
    uint32_t family_index = 0;
    for (const auto& queue : queue_properties) {
      if ((queue.queueFlags & config_->required_flags) &&
          queue.queueCount >= config_->queue_count) {
        return family_index;
      }
      ++family_index;
    }
    LOG(DEBUG) << "No suitable queue family found";
    return -1;
  }

  bool CheckExtensions(vk::PhysicalDevice device) {
    for (auto& [ext_name, is_available] : extension_availability_) {
      is_available = false;
    }
    auto available_ext = device.enumerateDeviceExtensionProperties();
    for (const auto& ext : available_ext) {
      extension_availability_[std::string(ext.extensionName)] = true;
    }
    for (const auto& [ext_name, is_available] : extension_availability_) {
      if (!is_available) {
        LOG(DEBUG) << "Extension " << ext_name << " is not available";
        return false;
      }
    }
    return true;
  }

  bool IsDeviceSuitable(vk::PhysicalDevice device) {
    return CheckFeatures(device) && CheckPresentModes(device) &&
           GetSuitableQueueFamilyIndex(device) != -1 && CheckExtensions(device);
  }

  uint64_t calcDeviceMemSize(vk::PhysicalDevice device) const {
    uint64_t res = 0;
    auto mem_properties = device.getMemoryProperties();
    for (const auto& heap : mem_properties.memoryHeaps) {
      res += heap.size;
    }
    return res;
  }

  bool DeviceCmp(vk::PhysicalDevice lhs, vk::PhysicalDevice rhs) const {
    if (!lhs) {
      return rhs;
    }
    if (!rhs) {
      return false;
    }
    uint32_t lhs_type =
        lhs.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
    uint32_t rhs_type =
        rhs.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
    if (lhs_type != rhs_type) {
      return lhs_type < rhs_type;
    }
    return calcDeviceMemSize(lhs) < calcDeviceMemSize(rhs);
  }

  void PickPhysicalDevice() {
    auto instance = Base::Get().GetInstance();
    auto physical_devices = instance.enumeratePhysicalDevices();
    for (auto& current_device : physical_devices) {
      LOG(DEBUG) << "Checking "
                 << std::string(current_device.getProperties().deviceName);

      if (IsDeviceSuitable(current_device) &&
          DeviceCmp(result_device_, current_device)) {
        result_device_ = current_device;
        result_queue_family_index_ =
            GetSuitableQueueFamilyIndex(result_device_);
      }
    }

    assert(result_device_);
    assert(result_queue_family_index_ != -1);

    LOG(DEBUG) << "Picked device: "
               << std::string(result_device_.getProperties().deviceName)
               << " With queue family: " << result_queue_family_index_;
  }

 public:
  PhysicalDevicePicker(const ContextConfig* config, vk::SurfaceKHR surface)
      : config_(config), surface_(surface) {
    for (const auto& ext_name : config->device_extensions) {
      extension_availability_[ext_name] = false;
    }
    PickPhysicalDevice();
  }

  vk::PhysicalDevice GetPickedDevice() const { return result_device_; }

  uint32_t GetQueueFamilyIndex() const { return result_queue_family_index_; }
};

}  // namespace

void Context::CreateWindow(ContextConfig& config) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window_ =
      glfwCreateWindow(config.window_extent.width, config.window_extent.height,
                       "RL2_WND_Name_Configure_TBD", nullptr, nullptr);
  assert(window_);
}

void Context::CreateSurface(ContextConfig& config) {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  auto res = glfwCreateWindowSurface(Base::Get().GetInstance(), window_,
                                     nullptr, &surface);
  assert(res == VK_SUCCESS);
  surface_ = vk::SurfaceKHR(surface);
}

void Context::PickPhysicalDevice(ContextConfig& config) {
  PhysicalDevicePicker picker(&config, surface_);
  physical_device_ = picker.GetPickedDevice();
  queue_family_index_ = picker.GetQueueFamilyIndex();
}

void Context::CreateDevice(ContextConfig& config) {
  std::vector<float> queue_priorities(config.queue_count, 1.0);
  vk::DeviceQueueCreateInfo queue_create_info(
      vk::DeviceQueueCreateFlags{}, queue_family_index_, queue_priorities);
  vk::PhysicalDeviceFeatures device_features;
  device_features.geometryShader = true;
  vk::DeviceCreateInfo device_info(vk::DeviceCreateFlags{}, queue_create_info,
                                   config.device_layers,
                                   config.device_extensions, &device_features);
  vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features(true);
  vk::PhysicalDeviceSynchronization2FeaturesKHR synchronization2_features(true);

  vk::StructureChain<vk::DeviceCreateInfo,
                     vk::PhysicalDeviceTimelineSemaphoreFeatures,
                     vk::PhysicalDeviceSynchronization2FeaturesKHR>
      info_chain(device_info, timeline_semaphore_features,
                 synchronization2_features);
  LOG(DEBUG) << "Creating device with:\nExt: " << config.device_extensions
             << "\nLayers: " << config.device_layers;
  device_ = physical_device_.createDevice(info_chain.get());
  assert(device_);

  device_queues_.resize(config.queue_count);
  for (uint32_t q_ind = 0; q_ind < device_queues_.size(); q_ind++) {
    device_queues_[q_ind] = device_.getQueue(queue_family_index_, q_ind);
    assert(device_queues_[q_ind]);
  }
}

Context::Context(ContextConfig config) {
  LOG(DEBUG) << "Creating window";
  CreateWindow(config);
  LOG(DEBUG) << "Creating surface";
  CreateSurface(config);
  LOG(DEBUG) << "Picking physical device";
  PickPhysicalDevice(config);
  LOG(DEBUG) << "Creating device";
  CreateDevice(config);
  LOG(DEBUG) << "Initialized context";
}

Context::Context(Context&& other) noexcept {
  Context tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Context::operator=(Context&& other) noexcept {
  Context tmp;
  tmp.Swap(other);
  Swap(tmp);
}

void Context::Swap(Context& other) noexcept {
  std::swap(device_, other.device_);
  std::swap(physical_device_, other.physical_device_);
  std::swap(queue_family_index_, other.queue_family_index_);
  device_queues_.swap(other.device_queues_);
  std::swap(window_, other.window_);
  std::swap(surface_, other.surface_);
}

Context::~Context() {
  if (!device_) {
    return;
  }
  auto instance = base::Base::Get().GetInstance();
  device_.waitIdle();
  device_.destroy();
  instance.destroySurfaceKHR(surface_);
  glfwDestroyWindow(window_);
}

}  // namespace base
