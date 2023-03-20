#include "common.hlsl"

groupshared uint l_key[KNElementsPerGroup];
groupshared uint l_pos[KNElementsPerGroup];
groupshared uint l_dig_hist[kNHistSize][2];

// Copy key and pos to fast local memory
void CopyToLocal(uint global_tidx, uint in_group_tidx) {
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_tidx * kNElementsPerThread + i;
    uint g_idx = global_tidx * kNElementsPerThread + i;
    bool is_in_range = g_idx < stage_info.n_elements;
    l_key[l_idx] = is_in_range ? g_key[g_idx] : (uint)-1;
    l_pos[l_idx] = is_in_range ? g_pos[g_idx] : (uint)-1;
    if (is_in_range && stage_info.bit_offset == 0) {
      l_pos[l_idx] = g_idx;
    }
  }
}

// Extract the bits, used for sorting at this stage and fill per-thread digit
// Count histogramm which is a 2d array N_BUCKETS X N_THREADS stored in a 
// Column-major order
void CalcDigHist(uint in_group_tidx) {
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_tidx * kNElementsPerThread + i;
    uint key_bucket = GetKeyBucket(l_key[l_idx]);
    l_dig_hist[key_bucket * KNThreadsPerGroup + in_group_tidx][0] += 1;
  }
}

void DighHistPrefSumStage(uint stage_offset, uint thread_idx, uint src_arr) {
  for (uint i = 0; i < kNHistBuckets; i++) {
    uint elem_idx = thread_idx * kNHistBuckets + i;
    uint val_to_add = elem_idx >= stage_offset ? l_dig_hist[elem_idx - stage_offset][src_arr] : 0;
    l_dig_hist[elem_idx][1 - src_arr] = l_dig_hist[elem_idx][src_arr] + val_to_add;
  }
}

void ClearDigHist(uint in_group_tidx) {
  for (uint i = 0; i < kNHistBuckets; i++) {
    l_dig_hist[in_group_tidx * kNHistBuckets + i][0] = 0;
    l_dig_hist[in_group_tidx * kNHistBuckets + i][1] = 0;
  }
}

// Compute prefix sums on l_dig_hist
uint CalcDigHistPrefSum(uint in_group_tidx) {
  uint src_array = 0;
  for (uint stage_offset = 1; stage_offset < kNHistSize; stage_offset *= 2) {
    DighHistPrefSumStage(stage_offset, in_group_tidx, src_array);
    GroupMemoryBarrierWithGroupSync();
    src_array = 1 - src_array;
  }
  // so it points to the array where final pref sums are stored
  return src_array;
}

[numthreads(64, 1, 1)]
void main(uint3 global_thread_idx : SV_DispatchThreadID,
          uint in_group_thread_idx : SV_GroupIndex,
          uint3 group_idx : SV_GroupID) {
  CopyToLocal(global_thread_idx.x, in_group_thread_idx);
  ClearDigHist(in_group_thread_idx);
  GroupMemoryBarrierWithGroupSync();

  CalcDigHist(in_group_thread_idx);
  GroupMemoryBarrierWithGroupSync();

  uint src_array = CalcDigHistPrefSum(in_group_thread_idx);

  uint dig_offsets[kNHistBuckets];
  for (uint bucket = 0; bucket < kNHistBuckets; bucket++) {
    uint idx = bucket * KNThreadsPerGroup + in_group_thread_idx;
    // pref-sums above are inclusive, but we need exclusive
    dig_offsets[bucket] = idx > 0 ? l_dig_hist[idx - 1][src_array] : 0;
  }

  GroupMemoryBarrierWithGroupSync();

  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint l_idx = in_group_thread_idx * kNElementsPerThread + i;
    uint key_bucket = GetKeyBucket(l_key[l_idx]);
    uint dst_idx = kNElementsPerThread * (global_thread_idx.x - in_group_thread_idx) + dig_offsets[key_bucket];
    dig_offsets[key_bucket] += 1;
    bool is_in_range = dst_idx < stage_info.n_elements;
    if (is_in_range) {
      g_key[dst_idx] = l_key[l_idx];
      g_pos[dst_idx] = l_pos[l_idx];
    }
  }

  if (in_group_thread_idx == KNThreadsPerGroup - 1) {
    for (uint dig = 0; dig < kNHistBuckets; dig++) {
      uint dig_count = dig_offsets[dig];
      if (dig > 0) {
        dig_count -= dig_offsets[dig - 1];
      }
      uint dst_idx = stage_info.n_groups * dig + group_idx.x;
      g_per_group_dig_hist[dst_idx] = dig_count;
    }
  }
}
