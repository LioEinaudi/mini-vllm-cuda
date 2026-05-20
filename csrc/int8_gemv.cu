/*
INT8 weight-only GEMV scaffold

Inputs:
  x:       [in_features], CUDA tensor, float16/bfloat16/float32
  w_int8:  [out_features, in_features], CUDA tensor, int8
  scales:  [out_features] or [out_features, 1], CUDA tensor

Output:
  y:       [out_features], float32 CUDA tensor

Math:
  w_dequant[o, i] = float(w_int8[o, i]) * scales[o]
  y[o] = sum_i x[i] * w_dequant[o, i]

Optimization roadmap:
  1. One output row per block with int8 vectorized loads.
  2. Accumulate in int32 or float depending on chosen quantization path.
  3. Add group-wise scales after the per-channel scaffold is stable.

Current version limits:
  Per-output-channel scale only. Uses ATen dequantize + matmul as a correctness
  placeholder and returns float32.
*/

#include "common.h"

torch::Tensor int8_gemv_cuda(torch::Tensor x, torch::Tensor w_int8, torch::Tensor scales) {
  MVC_CHECK_CUDA(x);
  MVC_CHECK_CUDA(w_int8);
  MVC_CHECK_CUDA(scales);
  MVC_CHECK_DTYPE(w_int8, torch::kInt8);
  TORCH_CHECK(x.dim() == 1, "x must have shape [in_features]");
  TORCH_CHECK(w_int8.dim() == 2, "w_int8 must have shape [out_features, in_features]");
  TORCH_CHECK(w_int8.size(1) == x.numel(), "in_features mismatch");

  auto x_float = x.to(torch::kFloat32);
  auto scale_float = scales.to(torch::kFloat32);
  if (scale_float.dim() == 1) {
    scale_float = scale_float.unsqueeze(1);
  }
  TORCH_CHECK(scale_float.size(0) == w_int8.size(0), "scales must match out_features");

  auto w_float = w_int8.to(torch::kFloat32) * scale_float;
  return torch::matmul(w_float, x_float);
}
