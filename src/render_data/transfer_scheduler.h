#pragma once

#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "render_graph/pass.h"
#include "utill/ring_buffer.h"

namespace render_data {

namespace {

class TransferRequest {
 protected:
  std::string dst_resource_name_;
  std::unique_ptr<char[]> data_;
  uint32_t transaction_ind_;

  void FillStagingBuffer(gpu_resources::Buffer& staging_buffer,
                         vk::DeviceSize& dst_offset,
                         vk::DeviceSize src_offset,
                         vk::DeviceSize size);

 public:
  TransferRequest() = default;
  TransferRequest(const std::string& dst_resource_name,
                  char* data,
                  vk::DeviceSize data_size,
                  uint32_t transaction_ind);

  TransferRequest(TransferRequest&& other) noexcept;
  void operator=(TransferRequest&& other) noexcept;
  void Swap(TransferRequest& other) noexcept;

  uint32_t GetTransactionInd() const;
  const std::string& GetDstResourceName() const;
};

class BufferTransferRequest : public TransferRequest {
  std::vector<vk::BufferCopy2KHR> copy_regions_;
  vk::DeviceSize total_size_;

 public:
  BufferTransferRequest() = default;
  BufferTransferRequest(const std::string& dst_buffer_name,
                        char* data,
                        vk::DeviceSize data_size,
                        std::vector<vk::BufferCopy2KHR> copy_regions,
                        uint32_t transaction_ind);

  BufferTransferRequest(BufferTransferRequest&& other) noexcept;
  void operator=(BufferTransferRequest&& other) noexcept;
  void Swap(BufferTransferRequest& other) noexcept;

  void RecordTransfer(vk::CommandBuffer cmd,
                      gpu_resources::Buffer& dst_buffer,
                      gpu_resources::Buffer& staging_buffer,
                      vk::DeviceSize& staging_offset);
};

struct ImageTransferRequest {
  std::string dst_image_name;
  std::unique_ptr<char[]> data;
  std::vector<vk::BufferImageCopy2KHR> copy_regions;
  uint32_t transaction_ind;
};

}  // namespace

class TransferScheduler : public render_graph::Pass {
  utill::RingBuffer<BufferTransferRequest> buffer_requests_;
  utill::RingBuffer<ImageTransferRequest> image_requests_;
  std::string staging_buffer_name_;
  uint32_t transaction_ind_ = 0;

 public:
  TransferScheduler(const std::string& staging_buffer_name,
                    const std::vector<std::string>& dst_buffers = {},
                    const std::vector<std::string>& dst_images = {},
                    uint32_t queue_max_size = 128);

  void IncrementTransactionInd();

  bool ScheduleBufferTransfer(const std::string& dst_buffer_name,
                              char* data,
                              vk::DeviceSize data_size,
                              std::vector<vk::BufferCopy2KHR> copy_regions);

  bool ScheduleImageTransfer(const std::string& dst_image_name,
                             char* data,
                             vk::DeviceSize data_size,
                             std::vector<vk::BufferImageCopy2KHR> copy_regions);
};

}  // namespace render_data
