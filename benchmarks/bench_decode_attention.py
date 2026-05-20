import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.benchmark_utils import benchmark_cuda


def main():
    if not mvc.is_extension_available():
        raise RuntimeError(f"CUDA extension is not available: {mvc.extension_error()!r}")

    seq_len = 1024
    num_heads = 32
    head_dim = 128
    q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

    latency_ms = benchmark_cuda(lambda: mvc.decode_attention(q, k_cache, v_cache, seq_len), warmup=10, iters=50)
    print(f"decode_attention: seq_len={seq_len} heads={num_heads} head_dim={head_dim} latency={latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
