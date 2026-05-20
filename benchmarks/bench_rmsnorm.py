import argparse

import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.benchmark_utils import benchmark_cuda_stats
from mini_vllm_cuda.torch_ref import rmsnorm_ref


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm latency with robust CUDA event stats.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup calls before timing.")
    parser.add_argument("--repeats", type=int, default=200, help="Number of independent timing samples.")
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=20,
        help="Calls measured inside each timing sample; each sample reports average ms per call.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not mvc.is_extension_available():
        raise RuntimeError(f"CUDA extension is not available: {mvc.extension_error()!r}")

    dtype = torch.float16
    eps = 1e-6
    num_tokens_list = [1, 8, 32, 128]
    hidden_sizes = [1024, 2048, 4096]

    print("RMSNorm latency, dtype=float16")
    print(
        f"methodology: warmup={args.warmup}, repeats={args.repeats}, "
        f"inner_iters={args.inner_iters}, reported latency=ms per call"
    )
    print(
        "num_tokens hidden_size "
        "torch_ref_median_ms torch_ref_p95_ms "
        "custom_median_ms custom_p95_ms "
        "speedup_median max_error mean_error"
    )

    for num_tokens in num_tokens_list:
        for hidden_size in hidden_sizes:
            torch.manual_seed(0)
            x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
            weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

            expected = rmsnorm_ref(x, weight, eps)
            actual = mvc.rmsnorm(x, weight, eps)
            error = (actual.float() - expected.float()).abs()
            max_error = error.max().item()
            mean_error = error.mean().item()

            ref_stats = benchmark_cuda_stats(
                lambda: rmsnorm_ref(x, weight, eps),
                warmup=args.warmup,
                repeats=args.repeats,
                inner_iters=args.inner_iters,
            )
            custom_stats = benchmark_cuda_stats(
                lambda: mvc.rmsnorm(x, weight, eps),
                warmup=args.warmup,
                repeats=args.repeats,
                inner_iters=args.inner_iters,
            )
            ref_median_ms = ref_stats["median_ms"]
            custom_median_ms = custom_stats["median_ms"]
            speedup = ref_median_ms / custom_median_ms if custom_median_ms > 0 else float("inf")

            print(
                f"{num_tokens:10d} {hidden_size:11d} "
                f"{ref_median_ms:19.4f} {ref_stats['p95_ms']:16.4f} "
                f"{custom_median_ms:16.4f} {custom_stats['p95_ms']:13.4f} "
                f"{speedup:14.2f} "
                f"{max_error:9.3e} {mean_error:10.3e}"
            )


if __name__ == "__main__":
    main()
