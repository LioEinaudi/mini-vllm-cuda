/*
RoPE CUDA kernel scaffold

Inputs:
  q:   [..., head_dim], CUDA tensor
  k:   [..., head_dim], CUDA tensor
  cos: broadcastable to q/k, usually [..., head_dim] or [head_dim]
  sin: broadcastable to q/k, usually [..., head_dim] or [head_dim]

Outputs:
  q_out, k_out: same shapes as q and k

Math:
  rotate_half([x1, x2]) = [-x2, x1]
  rope(x) = x * cos + rotate_half(x) * sin

Optimization roadmap:
  1. Map one vector row to one block or warp group.
  2. Use vectorized half/bfloat16 loads.
  3. Fuse q/k RoPE with cache write in a later decode path.

Current version limits:
  Uses half-split rotate_half semantics and ATen tensor ops as a correctness
  placeholder. head_dim must be even.
*/

#include "common.h"

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
