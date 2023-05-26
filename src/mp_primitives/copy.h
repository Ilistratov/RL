#pragma once

#include "gpu_resources/buffer.h"
#include "render_graph/pass.h"
#include "render_graph/render_graph.h"

#include <vector>

namespace mp_primitives {
namespace detail {

class CopyPass : public render_graph::Pass {
  vk::BufferCopy2 copy_region_ = {};
  gpu_resources::Buffer* src_ = nullptr;
  gpu_resources::Buffer* dst_ = nullptr;

 public:
  CopyPass() = default;
  CopyPass(gpu_resources::Buffer* src,
           gpu_resources::Buffer* dst,
           vk::BufferCopy2 copy_region);

  CopyPass(CopyPass const&) = delete;
  void operator=(CopyPass const&) = delete;

  CopyPass(CopyPass&& other) noexcept;
  void operator=(CopyPass&& other) noexcept;
  void Swap(CopyPass& other) noexcept;

  void OnPreRecord() override;
  void OnRecord(vk::CommandBuffer primary_cmd,
                const std::vector<vk::CommandBuffer>&) override;
};

class TinyCopy {
  CopyPass pass_;

 public:
  TinyCopy() = default;
  void Apply(render_graph::RenderGraph& render_graph,
             gpu_resources::Buffer* src,
             gpu_resources::Buffer* dst,
             vk::BufferCopy2 copy_region);
};

}  // namespace detail

using detail::TinyCopy;

}  // namespace mp_primitives
