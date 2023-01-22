#include "common.hlsl"

groupshared int warp_local[KNElementsPerGroup][2];

void CopyFromGlobalToLocal(uint global_idx, uint local_idx, uint dst_arr) {
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint idx = GetElemIdx(global_idx, i);
    int val = idx < stage_info.NElements ? val_arr[idx] : 0;
    warp_local[local_idx * kNElementsPerThread + i][dst_arr] = val;
  }
}

void DoScanStage(uint stage_offset, uint thread_idx, uint dst_arr) {
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint elem_idx = thread_idx * kNElementsPerThread + i;
    int val_to_add = elem_idx >= stage_offset ? warp_local[elem_idx - stage_offset][1 - dst_arr] : 0;
    warp_local[elem_idx][dst_arr] = warp_local[elem_idx][1 - dst_arr] + val_to_add;
  }
}

void CopyFromLocalToGlobal(uint global_idx, uint local_idx, uint src_arr) {
  for (uint i = 0; i < kNElementsPerThread; i++) {
    uint dst_idx = GetElemIdx(global_idx, i);
    int val = warp_local[local_idx * kNElementsPerThread + i][src_arr];
    if (dst_idx < stage_info.NElements) {
      val_arr[dst_idx] = val;
    }
  }
}

[numthreads(32, 1, 1)]
void main(uint3 global_idx : SV_DispatchThreadID, uint local_idx : SV_GroupIndex, uint3 group_idx : SV_GroupID) {
  CopyFromGlobalToLocal(global_idx.x, local_idx, 0);
  GroupMemoryBarrierWithGroupSync();

  uint dst_arr = 1;
  for (uint stage_offset = 1; stage_offset < KNElementsPerGroup; stage_offset *= 2, dst_arr = 1 - dst_arr) {
    DoScanStage(stage_offset, local_idx, dst_arr);
    GroupMemoryBarrierWithGroupSync();
  }

  CopyFromLocalToGlobal(global_idx.x, local_idx, 1 - dst_arr);
}
