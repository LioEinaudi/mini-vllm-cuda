#include <torch/extension.h>
#include "common.h"

#include <cmath>

static torch::Tensor rotate_half(torch::Tensor x) {
  int64_t dim = last_dim(x);
  TORCH_CHECK(dim % 2 == 0, "RoPE head_dim must be even");
  int64_t half = dim / 2;
  auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
  auto x2 = x.slice(/*dim=*/-1, /*start=*/half, /*end=*/dim);
  return torch::cat({-x2, x1}, /*dim=*/-1);
}

std::vector<torch::Tensor> rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin) {
  MVC_CHECK_CUDA(q);
  MVC_CHECK_CUDA(k);
  MVC_CHECK_CUDA(cos);
  MVC_CHECK_CUDA(sin);
  TORCH_CHECK(last_dim(q) == last_dim(k), "q and k must share head_dim");

  auto q_out = q * cos + rotate_half(q) * sin;
  auto k_out = k * cos + rotate_half(k) * sin;
  return {q_out, k_out};
}

torch::Tensor decode_attention_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t seq_len) {
  MVC_CHECK_CUDA(q);
  MVC_CHECK_CUDA(k_cache);
  MVC_CHECK_CUDA(v_cache);
  TORCH_CHECK(q.dim() == 2, "q must have shape [num_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 3, "k_cache must have shape [max_seq_len, num_heads, head_dim]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must match k_cache shape");
  TORCH_CHECK(seq_len > 0 && seq_len <= k_cache.size(0), "seq_len is out of range");
  TORCH_CHECK(q.size(0) == k_cache.size(1), "num_heads mismatch");
  TORCH_CHECK(q.size(1) == k_cache.size(2), "head_dim mismatch");

  auto qf = q.to(torch::kFloat32);
  auto kf = k_cache.narrow(/*dim=*/0, /*start=*/0, /*length=*/seq_len).to(torch::kFloat32);
  auto vf = v_cache.narrow(/*dim=*/0, /*start=*/0, /*length=*/seq_len).to(torch::kFloat32);
  double scale = 1.0 / std::sqrt(static_cast<double>(q.size(1)));

  auto scores = torch::einsum("hd,shd->hs", {qf, kf}) * scale;
  auto attn = torch::softmax(scores, /*dim=*/-1);
  auto out = torch::einsum("hs,shd->hd", {attn, vf});
  return out.to(q.scalar_type());
}

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
