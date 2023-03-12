#include "common.hlsl"

[[vk::binding(0, 2)]] RWStructuredBuffer<uint> g_dst_key;
[[vk::binding(1, 2)]] RWStructuredBuffer<uint> g_dst_pos;

[numthreads(64, 1, 1)]
void main(uint3 global_thread_idx : SV_DispatchThreadID, uint in_group_thread_idx : SV_GroupIndex, uint3 group_idx : SV_GroupID) {
  uint dst_bucket_start[kNHistBuckets];
  uint group_bucket_start[kNHistBuckets];
  for (uint i = 0; i < kNHistBuckets; i++) {
    uint idx = i * stage_info.n_groups + group_idx.x;
    dst_bucket_start[i] = idx > 0 ? g_per_group_dig_hist[idx - 1] : 0;
    group_bucket_start[i] = g_per_group_dig_hist[idx] - dst_bucket_start[i];
    group_bucket_start[i] += i > 0 ? group_bucket_start[i] : 0;
  }
  for (uint i = kNHistBuckets; i > 0; i--) {
    group_bucket_start[i] = group_bucket_start[i - 1];
  }
  group_bucket_start[0] = 0;

  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint src_idx = global_thread_idx.x * kNElementsPerThread + i;
    uint key = src_idx < stage_info.n_elements ? g_key[src_idx] : (uint)-1;
    uint pos = src_idx < stage_info.n_elements ? g_pos[src_idx] : (uint)-1;
    uint bucket = GetKeyBucket(key);
    uint dst_idx = dst_bucket_start[bucket] + in_group_thread_idx - group_bucket_start[bucket];
    if (dst_idx < stage_info.n_elements) {
      g_dst_key[dst_idx] = key;
      g_dst_key[dst_idx] = pos;
    }
  }
}