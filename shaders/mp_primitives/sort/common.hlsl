[[vk::binding(0, 0)]] RWStructuredBuffer<uint> g_key;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> g_pos;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> g_per_group_dig_hist;

const static uint kNBitPerPass = 4;
const static uint kNElementsPerThread = 8;
const static uint KNThreadsPerGroup = 64;
const static uint KNElementsPerGroup = kNElementsPerThread * KNThreadsPerGroup;
const static uint kNHistBuckets = 1U << kNBitPerPass;
const static uint kNHistSize = kNHistBuckets * KNThreadsPerGroup;

struct StageInformation {
  uint bit_offset;
  uint n_elements;
  uint n_groups;
};

[[vk::push_constant]] ConstantBuffer<StageInformation> stage_info;

uint GetKeyBucket(uint key) {
  return (key >> stage_info.bit_offset) & (kNHistBuckets - 1);
}
