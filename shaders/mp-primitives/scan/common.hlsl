[[vk::binding(0, 0)]] RWStructuredBuffer<int> val_arr;

struct StageInformation {
  uint StageSpacing;
  uint NElements;
};

[[vk::push_constant]] ConstantBuffer<StageInformation> stage_info;

const static uint kNElementsPerThread = 8;
const static uint KNThreadsPerGroup = 32;
const static uint KNElementsPerGroup = kNElementsPerThread * KNThreadsPerGroup;


uint GetElemIdx(uint thread_idx, uint elem_idx) {
  return (thread_idx * kNElementsPerThread + elem_idx + 1) * stage_info.StageSpacing - 1;
}
