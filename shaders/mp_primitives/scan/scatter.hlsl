#include "common.hlsl"

[numthreads(32, 1, 1)]
void main(uint3 global_idx : SV_DispatchThreadID, uint local_idx : SV_GroupIndex, uint3 group_idx : SV_GroupID) {
  uint prv_head_idx = (group_idx.x + 1) * KNElementsPerGroup * stage_info.StageSpacing - 1;
  int val_to_add = val_arr[prv_head_idx]; //Previous head
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint dst_idx = GetElemIdx(global_idx.x + KNThreadsPerGroup, i);
    if (dst_idx < stage_info.NElements && (dst_idx != prv_head_idx + stage_info.StageSpacing * KNElementsPerGroup)) {
      if (stage_info.IsSegmented) {
        val_to_add *= (1 - (int)head_flag[dst_idx]);
        head_flag[dst_idx] |= head_flag[prv_head_idx];
      }
      val_arr[dst_idx] += val_to_add;
    }
  }
}
