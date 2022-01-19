#pragma once

#include <map>

#include <vulkan/vulkan.hpp>

#include "gpu_resources/device_memory_allocator.h"

namespace gpu_resources {

template <typename UsageT>
class ResourceManagerBase {
 protected:
  // TODO switch to std::list<std::pair<uint32_t, UsageT>> as it makes more
  // sence here
  std::map<uint32_t, UsageT> usage_by_ind_;
  using UsageIt = typename std::map<uint32_t, UsageT>::const_iterator;

  UsageIt LoopedNext(UsageIt it) const {
    assert(it != usage_by_ind_.end());
    ++it;
    if (it == usage_by_ind_.end()) {
      it = usage_by_ind_.begin();
    }
    return it;
  }

  UsageIt LoopedPrev(UsageIt it) const {
    if (it == usage_by_ind_.begin()) {
      it = usage_by_ind_.end();
    }
    assert(it != usage_by_ind_.begin());
    --it;
    return it;
  }

  template <typename AdvanceFunc>
  UsageT GetDepChainUsage(UsageIt chain_start, AdvanceFunc advance_func) const {
    auto cur_it = advance_func(chain_start);
    auto nxt_it = advance_func(cur_it);
    UsageT result = cur_it->second;
    while (cur_it != chain_start) {
      result |= cur_it->second;
      if (cur_it->second.IsDependencyNeeded(nxt_it->second)) {
        break;
      }
      cur_it = nxt_it;
      nxt_it = advance_func(cur_it);
    }
    return result;
  }

  std::pair<UsageT, UsageT> GetUsageForBarrier(uint32_t user_ind) const {
    auto src_it = usage_by_ind_.find(user_ind);
    assert(src_it != usage_by_ind_.end());
    auto dst_it = LoopedNext(src_it);
    UsageT src_usage =
        GetDepChainUsage(dst_it, [this](UsageIt it) { return LoopedPrev(it); });
    UsageT dst_usage =
        GetDepChainUsage(src_it, [this](UsageIt it) { return LoopedNext(it); });
    return {src_usage, dst_usage};
  }

 public:
  // It is considered that accesses are perfomed sequentially in order of
  // user_ind increasments. Same order as order in which tasks in gpu_executer
  // are executed.
  void AddUsage(uint32_t user_ind, UsageT usage) {
    assert(!usage_by_ind_.contains(user_ind));
    usage_by_ind_[user_ind] = usage;
  }
};

}  // namespace gpu_resources
