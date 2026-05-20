import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.benchmark_utils import benchmark_cuda


def main():
    if not mvc.is_extension_available():
        raise RuntimeError(f"CUDA extension is not available: {mvc.extension_error()!r}")

    batch = 16
    num_heads = 32
    head_dim = 128
    q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(batch, 1, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(batch, 1, head_dim, device="cuda", dtype=torch.float16)

    latency_ms = benchmark_cuda(lambda: mvc.rope(q, k, cos, sin))
    print(f"rope: batch={batch} heads={num_heads} head_dim={head_dim} latency={latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
