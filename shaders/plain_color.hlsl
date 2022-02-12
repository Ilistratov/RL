[[vk::binding(0, 0)]] RWTexture2D<float4> outputImg;

struct PushConstants {
  uint width;
  uint height;
  float center_x;
  float center_y;
  float scale;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pushC;

float2 PixCordToWorldCord(uint2 pix_cord) {
  float2 uv =
      (float2(pix_cord) + float2(0.5, 0.5)) / uint2(pushC.width, pushC.height);
  uv = 2 * uv - float2(1, 1);
  float aspect = float(pushC.height) / pushC.width;
  float2 cord = float2(pushC.center_x, pushC.center_y) +
                (float2(pushC.scale, pushC.scale * aspect) * uv);
  return cord;
}

float3 Mandelbrot(float2 c) {
  float2 z = float2(0, 0);
  int iter_cnt = 0;
  while (iter_cnt < 128 && z.x * z.x + z.y * z.y < 4) {
    z = float2(z.x * z.x - z.y * z.y, 2 * z.x * z.y) + c;
    ++iter_cnt;
  }

  float val = iter_cnt / 128.0;

  if (iter_cnt == 128) {
    val = 0;
  }

  if (val < 0.5) {
    return float3(0, val * 2, 0);
  } else {
    return float3((val - 0.5) * 2, 1.0, (val - 0.5) * 2);
  }
}

[numthreads(8, 8, 1)] void main(uint3 DTid
                                : SV_DispatchThreadID, uint Gind
                                : SV_GroupIndex) {
  outputImg[DTid.xy] = float4(Mandelbrot(PixCordToWorldCord(DTid.xy)), 1.0);
}
