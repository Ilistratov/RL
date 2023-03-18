#include "gpu_resources/buffer.h"

#include "base/base.h"

#include <vulkan/vulkan_core.h>
#include "gpu_resources/common.h"
#include "gpu_resources/physical_buffer.h"
#include "utill/error_handling.h"

namespace gpu_resources {

using namespace error_messages;

Buffer::Buffer(BufferProperties properties, PassAccessSyncronizer* syncronizer)
    : syncronizer_(syncronizer), required_properties_(properties) {
  DCHECK(syncronizer_) << kErrSyncronizerNotProvided;
}

void Buffer::RequireProperties(BufferProperties properties) {
  required_properties_ =
      BufferProperties::Unite(required_properties_, properties);
}

void Buffer::DeclareAccess(ResourceAccess access, uint32_t pass_idx) const {
  DCHECK(syncronizer_) << kErrSyncronizerNotProvided;
  syncronizer_->AddAccess(buffer_, access, pass_idx);
}

void Buffer::RecordCopy(vk::CommandBuffer cmd,
                        const Buffer& src,
                        const Buffer& dst,
                        vk::DeviceSize src_offset,
                        vk::DeviceSize dst_offset,
                        vk::DeviceSize size) {
  RecordCopy(cmd, src, dst, {vk::BufferCopy2KHR(src_offset, dst_offset, size)});
}

void Buffer::RecordCopy(vk::CommandBuffer cmd,
                        const Buffer& src,
                        const Buffer& dst,
                        const std::vector<vk::BufferCopy2KHR>& copy_regions) {
  DCHECK(src.buffer_) << "src buffer: " << kErrResourceIsNull;
  DCHECK(src.buffer_->GetBuffer()) << "src buffer: " << kErrNotInitialized;
  DCHECK(dst.buffer_) << "dst buffer: " << kErrResourceIsNull;
  DCHECK(dst.buffer_->GetBuffer()) << "dst buffer: " << kErrNotInitialized;

  vk::CopyBufferInfo2KHR copy(src.buffer_->GetBuffer(),
                              dst.buffer_->GetBuffer(), copy_regions);
  cmd.copyBuffer2KHR(copy);
}

vk::DeviceSize Buffer::LoadDataFromPtr(void* data,
                                       vk::DeviceSize data_size,
                                       vk::DeviceSize dst_offset) {
  DCHECK(buffer_) << kErrNotInitialized;
  DCHECK(buffer_->GetBuffer()) << kErrNotInitialized;
  if (data_size == 0) {
    return dst_offset;
  }
  CHECK(dst_offset + data_size <= buffer_->GetSize()) << kErrNotEnoughSpace;

  void* mapping_start = buffer_->GetMappingStart();
  DCHECK(mapping_start) << kErrMemoryNotMapped;
  memcpy((char*)mapping_start + dst_offset, data, data_size);
  dst_offset += data_size;
  return dst_offset;
}

vk::Buffer Buffer::GetVkBuffer() const noexcept {
  DCHECK(buffer_) << kErrNotInitialized;
  return buffer_->GetBuffer();
}

PhysicalBuffer* Buffer::GetBuffer() const noexcept {
  return buffer_;
}

vk::DeviceSize Buffer::GetSize() const noexcept {
  DCHECK(buffer_) << kErrNotInitialized;
  return buffer_->GetSize();
}

Buffer::operator vk::Buffer() const noexcept {
  if (!buffer_) {
    return VK_NULL_HANDLE;
  }
  return buffer_->GetBuffer();
}

}  // namespace gpu_resources
