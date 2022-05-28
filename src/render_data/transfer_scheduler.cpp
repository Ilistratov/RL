#include "render_data/transfer_scheduler.h"

#include "utill/error_handling.h"

namespace render_data {

namespace {

void TransferRequest::FillStagingBuffer(gpu_resources::Buffer& staging_buffer,
                                        vk::DeviceSize& dst_offset,
                                        vk::DeviceSize src_offset,
                                        vk::DeviceSize size) {
  DCHECK(staging_buffer.GetSize() >= dst_offset + size)
      << "Out of staging buffer memory";
  staging_buffer.LoadDataFromPtr(data_.get() + src_offset, size, dst_offset);
  dst_offset += size;
}

TransferRequest::TransferRequest(const std::string& dst_resource_name,
                                 char* data,
                                 vk::DeviceSize data_size,
                                 uint32_t transaction_ind)
    : dst_resource_name_(dst_resource_name), transaction_ind_(transaction_ind) {
  DCHECK(data_size > 0) << "Size of data to transfer must be > 0";
  DCHECK(data) << "Pointer to transfer data must not be nullptr";
  data_ = std::make_unique<char[]>(data_size);
  memcpy(data_.get(), data, data_size);
}

TransferRequest::TransferRequest(TransferRequest&& other) noexcept {
  Swap(other);
}

void TransferRequest::operator=(TransferRequest&& other) noexcept {
  TransferRequest tmp(std::move(other));
  Swap(tmp);
}

void TransferRequest::Swap(TransferRequest& other) noexcept {
  dst_resource_name_.swap(other.dst_resource_name_);
  data_.swap(other.data_);
  std::swap(transaction_ind_, other.transaction_ind_);
}

uint32_t TransferRequest::GetTransactionInd() const {
  return transaction_ind_;
}

const std::string& TransferRequest::GetDstResourceName() const {
  return dst_resource_name_;
}

BufferTransferRequest::BufferTransferRequest(
    const std::string& dst_buffer_name,
    char* data,
    vk::DeviceSize data_size,
    std::vector<vk::BufferCopy2KHR> copy_regions,
    uint32_t transaction_ind)
    : TransferRequest(dst_buffer_name, data, data_size, transaction_ind),
      copy_regions_(copy_regions) {
  for (const auto& region : copy_regions_) {
    total_size_ += region.size;
  }
}

BufferTransferRequest::BufferTransferRequest(
    BufferTransferRequest&& other) noexcept {
  Swap(other);
}

void BufferTransferRequest::operator=(BufferTransferRequest&& other) noexcept {
  BufferTransferRequest tmp(std::move(other));
  Swap(tmp);
}
void BufferTransferRequest::Swap(BufferTransferRequest& other) noexcept {
  TransferRequest::Swap(other);
  copy_regions_.swap(other.copy_regions_);
}

void BufferTransferRequest::RecordTransfer(
    vk::CommandBuffer cmd,
    gpu_resources::Buffer& dst_buffer,
    gpu_resources::Buffer& staging_buffer,
    vk::DeviceSize& staging_offset) {
  for (auto& region : copy_regions_) {
    vk::DeviceSize src_offset = staging_offset;
    staging_buffer.LoadDataFromPtr(data_.get() + region.srcOffset, region.size,
                                   staging_offset);
    region.srcOffset = src_offset;
  }
  gpu_resources::Buffer::RecordCopy(cmd, staging_buffer, dst_buffer,
                                    copy_regions_);
}

}  // namespace

TransferScheduler::TransferScheduler(
    const std::string& staging_buffer_name,
    const std::vector<std::string>& dst_buffers,
    const std::vector<std::string>& dst_images,
    uint32_t queue_max_size)
    : buffer_requests_(queue_max_size),
      image_requests_(queue_max_size),
      staging_buffer_name_(staging_buffer_name) {
  gpu_resources::ResourceUsage buffer_transfer_dst_usage{
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferWrite, vk::ImageLayout::eUndefined};
  for (const auto& buffer_name : dst_buffers) {
    AddBuffer(buffer_name, render_graph::BufferPassBind(
                               buffer_transfer_dst_usage,
                               vk::BufferUsageFlagBits::eTransferDst));
  }

  gpu_resources::ResourceUsage image_transfer_dst_usage =
      buffer_transfer_dst_usage;
  image_transfer_dst_usage.layout = vk::ImageLayout::eTransferDstOptimal;
  for (const auto& image_name : dst_images) {
    AddImage(image_name,
             render_graph::ImagePassBind(image_transfer_dst_usage,
                                         vk::ImageUsageFlagBits::eTransferDst));
  }

  gpu_resources::ResourceUsage staging_buffer_usage{
      vk::PipelineStageFlagBits2KHR::eTransfer,
      vk::AccessFlagBits2KHR::eTransferRead, vk::ImageLayout::eUndefined};
  AddBuffer(staging_buffer_name,
            render_graph::BufferPassBind(
                staging_buffer_usage, vk::BufferUsageFlagBits::eTransferSrc));
}

void TransferScheduler::IncrementTransactionInd() {
  ++transaction_ind_;
}

bool TransferScheduler::ScheduleBufferTransfer(
    const std::string& dst_buffer_name,
    char* data,
    vk::DeviceSize data_size,
    std::vector<vk::BufferCopy2KHR> copy_regions) {
  if (buffer_requests_.IsFull()) {
    return false;
  }
  BufferTransferRequest request(dst_buffer_name, data, data_size, copy_regions,
                                transaction_ind_);
  buffer_requests_.PushBack(std::move(request));
  return true;
}

}  // namespace render_data