#include "gpu_resources/memory_block.h"

#include <cassert>

namespace gpu_resources {

vk::DeviceSize MemoryBlock::GetAlignedOffset(vk::DeviceSize alignment) const {
  vk::DeviceSize offset_diff = alignment - offset % alignment;
  if (offset_diff == alignment) {
    offset_diff = 0;
  }
  return offset + offset_diff;
}

MemoryBlock MemoryBlock::Suballocate(vk::DeviceSize block_size,
                                     vk::DeviceSize alignment) {
  vk::DeviceSize n_offset = GetAlignedOffset(alignment);
  assert(n_offset + block_size < size);
  MemoryBlock result = {memory, block_size, n_offset, type_index};
  offset = n_offset + block_size;
  return result;
}

}  // namespace gpu_resources
