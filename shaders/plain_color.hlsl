[[vk::binding(0, 0)]] RWTexture2D<float4> outputImg;

[numthreads(8, 8, 1)] void main(uint3 DTid
                                : SV_DispatchThreadID, uint Gind
                                : SV_GroupIndex) {
  outputImg[DTid.xy] = float4(1.0, 0.0, 1.0, 1.0);
}
