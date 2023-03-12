#include "common.hlsl"

groupshared uint l_key[KNElementsPerGroup];
groupshared uint l_pos[KNElementsPerGroup];
groupshared uint l_dig_hist[kNHistSize][2];

[numthreads(64, 1, 1)]
void main(uint3 global_thread_idx : SV_DispatchThreadID, uint in_group_thread_idx : SV_GroupIndex, uint3 group_idx : SV_GroupID) {
  // Copy key and pos to fast local memory
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_thread_idx * kNElementsPerThread + i;
    uint g_idx = global_thread_idx.x * kNElementsPerThread + i;
    bool is_in_range = g_idx < stage_info.n_elements;
    l_key[l_idx] = is_in_range ? g_key[g_idx] : (uint)-1;
    l_pos[l_idx] = is_in_range ? g_pos[g_idx] : (uint)-1;
    if (is_in_range && stage_info.bit_offset == 0) {
      l_key = g_idx;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // Extract the bits, used for sorting at this stage and fill per-thread digit
  // Count histogramm which is a 2d array N_BUCKETS X N_THREADS stored in a 
  // Column-major order
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_thread_idx * kNElementsPerThread + i;
    uint key_bucket = GetKeyBucket(l_key[l_idx]);
    l_dig_hist[key_bucket * KNThreadsPerGroup + in_group_thread_idx][0] += 1;
  }
  GroupMemoryBarrierWithGroupSync();

  // Compute prefix sums on l_dig_hist
  uint src_array = 0;
  for (uint stage_offset = 0; stage_offset < kNHistSize; stage_offset <<= 1) {
    for (uint i = 0; i < kNHistBuckets; i++) {
      uint dst_el = in_group_thread_idx * kNElementsPerThread + i;
      uint val_to_add = dst_el >= stage_offset ? l_dig_hist[dst_el - stage_offset][src_array] : 0;
      l_dig_hist[dst_el][1 - src_array] = l_dig_hist[dst_el][src_array] + val_to_add;
    }
    src_array ^= 1;
    GroupMemoryBarrierWithGroupSync();
  }
  src_array ^= 1; // so it points to the array where final pref sums are stored

  uint dig_offsets[kNHistBuckets];
  for (uint i = 0; i < kNHistBuckets; i++) {
    uint idx = i * KNThreadsPerGroup + in_group_thread_idx;
    // pref-sums above are inclusive, but we need exclusive
    dig_offsets[i] = idx > 0 ? l_dig_hist[idx - 1][src_array] : 0;
  }

  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_thread_idx * kNElementsPerThread + i;
    uint key_bucket = GetKeyBucket(l_key[l_idx]);
    uint dst_idx = dig_offsets[key_bucket] + global_thread_idx.x - in_group_thread_idx;
    dig_offsets[key_bucket] += 1;
    bool is_in_range = dst_idx < stage_info.n_elements;
    if (is_in_range) {
      g_key[dst_idx] = l_key[l_idx];
      g_pos[dst_idx] = l_pos[l_idx];
    }
  }

  if (in_group_thread_idx == KNThreadsPerGroup - 1) {
    for (uint i = 0; i < kNHistBuckets; i++) {
      if (i > 0) {
        // In last thread of group doing so will result in dig_offsets having
        // Digit counts for the processed keys
        dig_offsets[i] -= dig_offsets[i - 1];
      }
      uint dst_idx = stage_info.n_groups * i + group_idx.x;
      g_per_group_dig_hist[dst_idx] = dig_offsets[i];
    }
  }
}