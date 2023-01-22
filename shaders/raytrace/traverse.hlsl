#include "common.hlsl"
#include "bvh.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<RayTraversalState> traversal_state_out;
[[vk::binding(1, 0)]] StructuredBuffer<float4> vertex_pos;
[[vk::binding(2, 0)]] StructuredBuffer<float4> vertex_normal;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> vertex_ind;
[[vk::binding(4, 0)]] StructuredBuffer<BVHNode> bvh_buffer;



[numthreads(16, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint Gind : SV_GroupIndex) {
}
