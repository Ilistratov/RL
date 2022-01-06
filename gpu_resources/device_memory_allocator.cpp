#include "gpu_resources/device_memory_allocator.h"

#include "base/base.h"

namespace gpu_resources {

bool DeviceMemoryAllocator::IsMemoryTypeSuitable(
    uint32_t type_index,
    uint32_t type_bits,
    vk::MemoryPropertyFlags property_flags) const {
  if ((type_bits & (1u << type_index)) == 0) {
    return false;
  }
  auto available_poperty_flags =
      property_flags &
      device_memory_properties_.memoryTypes[type_index].propertyFlags;
  return available_poperty_flags == property_flags;
}

DeviceMemoryAllocator::DeviceMemoryAllocator() {
  device_memory_properties_ =
      base::Base::Get().GetContext().GetPhysicalDevice().getMemoryProperties();
}

uint32_t DeviceMemoryAllocator::FindMemoryTypeIndex(
    vk::MemoryRequirements requierments,
    vk::MemoryPropertyFlags property_flags) const {
  for (uint32_t type_index = 0;
       type_index < device_memory_properties_.memoryTypeCount; type_index++) {
    if (IsMemoryTypeSuitable(type_index, requierments.memoryTypeBits,
                             property_flags)) {
      return type_index;
    }
  }

  return UINT32_MAX;
}

void DeviceMemoryAllocator::AddMemoryBlock(
    vk::MemoryRequirements requierments,
    vk::MemoryPropertyFlags property_flags) {
  uint32_t type_index = FindMemoryTypeIndex(requierments, property_flags);
  assert(type_index != UINT32_MAX);
  auto new_block_size =
      memory_by_type_ind_[type_index].GetAlignedOffset(requierments.alignment);
  new_block_size += requierments.size;
  memory_by_type_ind_[type_index].offset = new_block_size;
}

void DeviceMemoryAllocator::Allocate() {
  auto device = base::Base::Get().GetContext().GetDevice();
  for (auto& [type_ind, block] : memory_by_type_ind_) {
    block.size = block.offset;
    block.offset = 0;
    block.type_index = type_ind;
    block.memory = device.allocateMemory(
        vk::MemoryAllocateInfo{block.size, block.type_index});
  }
}

MemoryBlock DeviceMemoryAllocator::GetMemoryBlock(
    vk::MemoryRequirements requierments,
    vk::MemoryPropertyFlags property_flags) {
  uint32_t type_index = FindMemoryTypeIndex(requierments, property_flags);
  assert(type_index != UINT32_MAX);
  assert(memory_by_type_ind_.contains(type_index));
  return memory_by_type_ind_[type_index].Suballocate(requierments.size,
                                                     requierments.alignment);
}

DeviceMemoryAllocator::~DeviceMemoryAllocator() {
  auto device = base::Base::Get().GetContext().GetDevice();
  for (auto& [type_ind, block] : memory_by_type_ind_) {
    device.freeMemory(block.memory);
  }
}

}  // namespace gpu_resources
