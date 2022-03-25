#include "gpu_resources/device_memory_allocator.h"

#include "base/base.h"

#include "utill/error_handling.h"

namespace gpu_resources {

uint32_t DeviceMemoryAllocator::GetSuitableTypeBits(
    vk::MemoryRequirements requierments,
    vk::MemoryPropertyFlags property_flags) const {
  uint32_t result = requierments.memoryTypeBits;
  for (uint32_t type_index = 0;
       type_index < device_memory_properties_.memoryTypeCount; type_index++) {
    if ((result & (1u << type_index)) == 0) {
      continue;
    }
    auto available_poperty_flags =
        property_flags &
        device_memory_properties_.memoryTypes[type_index].propertyFlags;
    if (available_poperty_flags != property_flags) {
      result ^= (1u << type_index);
    }
  }
  return result;
}

uint32_t DeviceMemoryAllocator::FindTypeIndex(uint32_t type_bits) const {
  uint32_t result = -1;
  for (uint32_t type_index = 0;
       type_index < device_memory_properties_.memoryTypeCount; type_index++) {
    if (((1u << type_index) & type_bits) == 0) {
      continue;
    }
    if (memory_by_type_ind_.contains(type_index)) {
      result = type_index;
      break;
    }
    if (result == uint32_t(-1)) {
      result = type_index;
    }
  }
  return result;
}

void DeviceMemoryAllocator::ExtendPreallocBlock(uint32_t type_index,
                                                vk::DeviceSize alignment,
                                                vk::DeviceSize size) {
  assert(type_index < device_memory_properties_.memoryTypeCount);
  assert(size > 0);
  auto new_block_size =
      memory_by_type_ind_[type_index].GetAlignedOffset(alignment);
  new_block_size += size;
  memory_by_type_ind_[type_index].offset = new_block_size;
}

DeviceMemoryAllocator::DeviceMemoryAllocator() {
  device_memory_properties_ =
      base::Base::Get().GetContext().GetPhysicalDevice().getMemoryProperties();
}

void DeviceMemoryAllocator::Allocate() {
  auto device = base::Base::Get().GetContext().GetDevice();
  for (auto& [type_ind, block] : memory_by_type_ind_) {
    block.size = block.offset;
    block.offset = 0;
    block.type_index = type_ind;
    auto alloc_result = device.allocateMemory(
        vk::MemoryAllocateInfo{block.size, block.type_index});
    CHECK_VK_RESULT(alloc_result.result) << "Failed to allocate device memory.";
    block.memory = alloc_result.value;
  }
  for (auto& allocation : allocations_) {
    // info for actual allocation stored in RequestMemory is used here
    allocation = memory_by_type_ind_[allocation.type_index].Suballocate(
        allocation.size, allocation.offset);
  }
}

MemoryBlock* DeviceMemoryAllocator::RequestMemory(
    vk::MemoryRequirements requierments,
    vk::MemoryPropertyFlags property_flags) {
  uint32_t type_bits = GetSuitableTypeBits(requierments, property_flags);
  uint32_t type_index = FindTypeIndex(type_bits);
  ExtendPreallocBlock(type_index, requierments.alignment, requierments.size);
  // store info for actual allocation for future use
  MemoryBlock result;
  result.type_index = type_index;
  result.offset = requierments.alignment;
  result.size = requierments.size;
  allocations_.push_back(result);
  return &allocations_.back();
}

DeviceMemoryAllocator::~DeviceMemoryAllocator() {
  auto device = base::Base::Get().GetContext().GetDevice();
  for (auto& [type_ind, block] : memory_by_type_ind_) {
    device.freeMemory(block.memory);
  }
}

}  // namespace gpu_resources
