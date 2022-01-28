[[vk::binding(0, 0)]] RWTexture2D<float4> outputImg;

struct PushConstants {
  uint s_width;
  uint s_height;
  uint tm_milisec;
  float mouse_x;
  float mouse_y;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

void DrawGrid(uint2 pix_pos, uint grid_dim, float3 col) {
  if ((pix_pos.x & (grid_dim - 1)) == 0 || (pix_pos.y & (grid_dim - 1)) == 0) {
    outputImg[pix_pos.xy] = float4(col, 1.0);
  }
}

[numthreads(8, 8, 1)] void main(uint3 DTid
                                : SV_DispatchThreadID, uint Gind
                                : SV_GroupIndex) {
  float mouse_dst = distance(
      float2(pushConstants.mouse_x, pushConstants.mouse_y), float2(DTid.xy));
  float max_dst = length(float2(pushConstants.s_width, pushConstants.s_height));
  float val = 1 - mouse_dst / max_dst;
  val = pow(val, 4);
  outputImg[DTid.xy] = float4(val, val, val, 1.0);
  DrawGrid(DTid.xy, 16, float3(0.3, 0, 0));
  DrawGrid(DTid.xy, 32, float3(0.6, 0, 0));
  DrawGrid(DTid.xy, 64, float3(0.9, 0, 0));
}
