import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.benchmark_utils import benchmark_cuda


def main():
    if not mvc.is_extension_available():
        raise RuntimeError(f"CUDA extension is not available: {mvc.extension_error()!r}")

    in_features = 4096
    out_features = 4096
    x = torch.randn(in_features, device="cuda", dtype=torch.float16)
    w_int8 = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
    scales = torch.rand(out_features, device="cuda", dtype=torch.float16) * 0.02

    latency_ms = benchmark_cuda(lambda: mvc.int8_gemv(x, w_int8, scales), warmup=10, iters=50)
    print(f"int8_gemv: in={in_features} out={out_features} latency={latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
