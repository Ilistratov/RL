#include "base/physical_device_picker.h"

#include "base/base.h"
#include "utill/logger.h"

namespace base {

bool PhysicalDevicePicker::CheckFeatures(vk::PhysicalDevice device) const {
  return device.getFeatures().geometryShader;
}

bool PhysicalDevicePicker::CheckPresentModes(vk::PhysicalDevice device) const {
  return !device.getSurfacePresentModesKHR(surface_).empty() &&
         !device.getSurfaceFormatsKHR(surface_).empty();
}

uint32_t PhysicalDevicePicker::GetSuitableQueueFamilyIndex(
    vk::PhysicalDevice device) const {
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

bool PhysicalDevicePicker::CheckExtensions(vk::PhysicalDevice device) {
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

bool PhysicalDevicePicker::CheckSurfaceSupport(vk::PhysicalDevice device) {
  return device.getSurfaceSupportKHR(GetSuitableQueueFamilyIndex(device),
                                     surface_);
}

bool PhysicalDevicePicker::IsDeviceSuitable(vk::PhysicalDevice device) {
  return CheckFeatures(device) && CheckPresentModes(device) &&
         GetSuitableQueueFamilyIndex(device) != uint32_t(-1) &&
         CheckExtensions(device) && CheckSurfaceSupport(device);
}

uint64_t PhysicalDevicePicker::calcDeviceMemSize(
    vk::PhysicalDevice device) const {
  uint64_t res = 0;
  auto mem_properties = device.getMemoryProperties();
  for (const auto& heap : mem_properties.memoryHeaps) {
    res += heap.size;
  }
  return res;
}

bool PhysicalDevicePicker::DeviceCmp(vk::PhysicalDevice lhs,
                                     vk::PhysicalDevice rhs) const {
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

void PhysicalDevicePicker::PickPhysicalDevice() {
  auto instance = Base::Get().GetInstance();
  auto physical_devices = instance.enumeratePhysicalDevices();
  for (auto& current_device : physical_devices) {
    LOG(DEBUG) << "Checking "
               << std::string(current_device.getProperties().deviceName);

    if (IsDeviceSuitable(current_device) &&
        DeviceCmp(result_device_, current_device)) {
      result_device_ = current_device;
      result_queue_family_index_ = GetSuitableQueueFamilyIndex(result_device_);
    }
  }

  assert(result_device_);
  assert(result_queue_family_index_ != uint32_t(-1));

  LOG(DEBUG) << "Picked device: "
             << std::string(result_device_.getProperties().deviceName)
             << " With queue family: " << result_queue_family_index_;
}

PhysicalDevicePicker::PhysicalDevicePicker(const ContextConfig* config,
                                           vk::SurfaceKHR surface)
    : config_(config), surface_(surface) {
  for (const auto& ext_name : config->device_extensions) {
    extension_availability_[ext_name] = false;
  }
  PickPhysicalDevice();
}

vk::PhysicalDevice PhysicalDevicePicker::GetPickedDevice() const {
  return result_device_;
}

uint32_t PhysicalDevicePicker::GetQueueFamilyIndex() const {
  return result_queue_family_index_;
}

}  // namespace base
