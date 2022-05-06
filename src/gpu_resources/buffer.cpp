#include "gpu_resources/buffer.h"

#include "base/base.h"

#include "utill/error_handling.h"

namespace gpu_resources {

Buffer::Buffer(vk::DeviceSize size, vk::MemoryPropertyFlags memory_flags)
    : size_(size), memory_flags_(memory_flags) {}

void Buffer::AddUsage(uint32_t user_ind,
                      ResourceUsage usage,
                      vk::BufferUsageFlags buffer_usage_flags) {
  DCHECK(!buffer_) << "Can't use this method after VkBuffer was created";
  usage.layout = vk::ImageLayout::eUndefined;
  access_manager_.AddUsage(user_ind, usage);
  usage_flags_ |= buffer_usage_flags;
}

void Buffer::CreateVkBuffer() {
  DCHECK(!buffer_) << "VkBuffer already created";
  DCHECK(size_ > 0) << "Can't create empty buffer";
  auto device = base::Base::Get().GetContext().GetDevice();
  buffer_ = device.createBuffer(vk::BufferCreateInfo(
      {}, size_, usage_flags_, vk::SharingMode::eExclusive, {}));
}

void Buffer::SetDebugName(const std::string& debug_name) const {
  DCHECK(buffer_) << "Resource must be created to use this method";
  auto device = base::Base::Get().GetContext().GetDevice();
  device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
      buffer_.objectType, (uint64_t)(VkBuffer)buffer_, debug_name.c_str()));
}

void Buffer::RequestMemory(DeviceMemoryAllocator& allocator) {
  DCHECK(buffer_) << "VkBuffer must be created to use this method";
  DCHECK(!memory_) << "Memory already requested";
  auto device = base::Base::Get().GetContext().GetDevice();
  auto mem_requierments = device.getBufferMemoryRequirements(buffer_);
  memory_ = allocator.RequestMemory(mem_requierments, memory_flags_);
}

vk::BindBufferMemoryInfo Buffer::GetBindMemoryInfo() const {
  DCHECK(buffer_) << "VkBuffer must be created to use this method";
  DCHECK(memory_) << "Memory must be requested to use this method";
  DCHECK(memory_->memory)
      << "Requested memory must be allocated to use this method";
  return vk::BindBufferMemoryInfo(buffer_, memory_->memory, memory_->offset);
}

Buffer::Buffer(Buffer&& other) noexcept {
  Swap(other);
}

void Buffer::operator=(Buffer&& other) noexcept {
  Buffer tmp(std::move(other));
  Swap(tmp);
}
void Buffer::Swap(Buffer& other) noexcept {
  std::swap(access_manager_, other.access_manager_);
  std::swap(buffer_, other.buffer_);
  std::swap(size_, other.size_);
  std::swap(usage_flags_, other.usage_flags_);
  std::swap(memory_flags_, other.memory_flags_);
  std::swap(memory_, other.memory_);
}

Buffer::~Buffer() {
  auto device = base::Base::Get().GetContext().GetDevice();
  device.destroyBuffer(buffer_);
}

vk::Buffer Buffer::GetBuffer() const {
  return buffer_;
}

vk::DeviceSize Buffer::GetSize() const {
  return size_;
}

void* Buffer::GetMappingStart() const {
  DCHECK(memory_) << "Memory must be requested to use this method";
  return memory_->mapping_start;
}

vk::MappedMemoryRange Buffer::GetMappedMemoryRange() const {
  DCHECK(memory_) << "Memory must be requested to use this method";
  return vk::MappedMemoryRange(memory_->memory, memory_->offset, memory_->size);
}

vk::BufferMemoryBarrier2KHR Buffer::GetBarrier(
    vk::PipelineStageFlags2KHR src_stage_flags,
    vk::AccessFlags2KHR src_access_flags,
    vk::PipelineStageFlags2KHR dst_stage_flags,
    vk::AccessFlags2KHR dst_access_flags) const {
  return vk::BufferMemoryBarrier2KHR(src_stage_flags, src_access_flags,
                                     dst_stage_flags, dst_access_flags, {}, {},
                                     buffer_, 0, size_);
}
vk::BufferMemoryBarrier2KHR Buffer::GetPostPassBarrier(uint32_t user_ind) {
  auto [src_usage, dst_usage] = access_manager_.GetUserDeps(user_ind);
  if (src_usage.stage == vk::PipelineStageFlagBits2KHR::eNone &&
      dst_usage.stage == vk::PipelineStageFlagBits2KHR::eNone) {
    return {};
  }
  return GetBarrier(src_usage.stage, src_usage.access, dst_usage.stage,
                    dst_usage.access);
}

void Buffer::RecordCopy(vk::CommandBuffer cmd,
                        const Buffer& src,
                        const Buffer& dst,
                        vk::DeviceSize src_offset,
                        vk::DeviceSize dst_offset,
                        vk::DeviceSize size) {
  RecordCopy(cmd, src, dst, {vk::BufferCopy(src_offset, dst_offset, size)});
}

void Buffer::RecordCopy(vk::CommandBuffer cmd,
                        const Buffer& src,
                        const Buffer& dst,
                        const std::vector<vk::BufferCopy>& copy_regions) {
  DCHECK(src.buffer_) << "src must be created to use this method";
  DCHECK(dst.buffer_) << "dst must be created to use this method";
  cmd.copyBuffer(src.buffer_, dst.buffer_, copy_regions);
}

vk::DeviceSize Buffer::LoadDataFromPtr(void* data,
                                       vk::DeviceSize data_size,
                                       vk::DeviceSize dst_offset) {
  if (data_size == 0) {
    return dst_offset;
  }
  if (dst_offset + data_size > size_) {
    return -1;
  }
  DCHECK(memory_) << "Memory not bound";
  DCHECK(memory_->mapping_start) << "Memory not mapped";
  memcpy((char*)memory_->mapping_start + dst_offset, data, data_size);
  dst_offset += data_size;
  return dst_offset;
}

}  // namespace gpu_resources
