#ifndef SORT_COMMON
#define SORT_COMMON

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> g_key;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> g_pos;
[[vk::binding(0, 1)]] RWStructuredBuffer<uint> g_per_group_dig_hist;

struct StageInformation {
  uint n_elements;
  uint bit_offset;
  uint n_groups;
};

[[vk::push_constant]] ConstantBuffer<StageInformation> stage_info;

const static uint kNBitPerPass = 4;
const static uint kNElementsPerThread = 16;
const static uint KNThreadsPerGroup = 64;
const static uint KNElementsPerGroup = kNElementsPerThread * KNThreadsPerGroup;
const static uint kNHistBuckets = 1U << kNBitPerPass;
const static uint kNHistSize = kNHistBuckets * KNThreadsPerGroup;

uint GetKeyBucket(uint key) {
  return (key >> stage_info.bit_offset) & (kNHistBuckets - 1);
}

#endif // SORT_COMMON