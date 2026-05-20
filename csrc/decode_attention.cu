/*
Decode attention with continuous KV cache scaffold

Inputs:
  q:       [num_heads, head_dim], CUDA tensor for one decode token
  k_cache: [max_seq_len, num_heads, head_dim], CUDA tensor
  v_cache: [max_seq_len, num_heads, head_dim], CUDA tensor
  seq_len: number of valid cache tokens to attend to

Output:
  out:     [num_heads, head_dim], CUDA tensor

Math:
  scores[h, s] = dot(q[h], k_cache[s, h]) / sqrt(head_dim)
  attn[h, s] = softmax(scores[h, :])
  out[h] = sum_s attn[h, s] * v_cache[s, h]

Optimization roadmap:
  1. One head per block for small head_dim.
  2. Online softmax over sequence blocks to avoid materializing scores.
  3. Extend from continuous cache to paged KV cache indexing.

Current version limits:
  Continuous cache only. seq_len must be <= max_seq_len. Uses ATen einsum as a
  correctness placeholder.
*/

#include "common.h"

#include <cmath>

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
