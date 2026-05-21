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
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="Run one num_tokens value instead of the default sweep.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Run one hidden_size value instead of the default sweep.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Input dtype to benchmark.",
    )
    parser.add_argument(
        "--impl",
        choices=("v0", "v1", "all"),
        default="v0",
        help="Custom CUDA RMSNorm implementation to benchmark.",
    )
    args = parser.parse_args()
    if args.num_tokens is not None and args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    if args.hidden_size is not None and args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    return args


def main():
    args = parse_args()
    if not mvc.is_extension_available():
        raise RuntimeError(f"CUDA extension is not available: {mvc.extension_error()!r}")

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    eps = 1e-6
    num_tokens_list = [args.num_tokens] if args.num_tokens is not None else [1, 8, 32, 128]
    hidden_sizes = [args.hidden_size] if args.hidden_size is not None else [1024, 2048, 4096]
    impls = {
        "v0": mvc.rmsnorm,
        "v1": mvc.rmsnorm_v1,
    }
    selected_impls = impls.items() if args.impl == "all" else [(args.impl, impls[args.impl])]

    print(f"RMSNorm latency, dtype={args.dtype}, impl={args.impl}")
    print(
        f"methodology: warmup={args.warmup}, repeats={args.repeats}, "
        f"inner_iters={args.inner_iters}, reported latency=ms per call"
    )
    print(
        "impl num_tokens hidden_size "
        "torch_ref_median_ms torch_ref_p95_ms "
        "custom_median_ms custom_p95_ms "
        "speedup_median max_error mean_error"
    )

    for num_tokens in num_tokens_list:
        for hidden_size in hidden_sizes:
            torch.manual_seed(0)
            x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
            weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

            ref_stats = benchmark_cuda_stats(
                lambda: rmsnorm_ref(x, weight, eps),
                warmup=args.warmup,
                repeats=args.repeats,
                inner_iters=args.inner_iters,
            )
            ref_median_ms = ref_stats["median_ms"]

            for impl_name, impl_fn in selected_impls:
                expected = rmsnorm_ref(x, weight, eps)
                actual = impl_fn(x, weight, eps)
                error = (actual.float() - expected.float()).abs()
                max_error = error.max().item()
                mean_error = error.mean().item()

                custom_stats = benchmark_cuda_stats(
                    lambda fn=impl_fn: fn(x, weight, eps),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    inner_iters=args.inner_iters,
                )
                custom_median_ms = custom_stats["median_ms"]
                speedup = ref_median_ms / custom_median_ms if custom_median_ms > 0 else float("inf")

                print(
                    f"{impl_name:>4s} {num_tokens:10d} {hidden_size:11d} "
                    f"{ref_median_ms:19.4f} {ref_stats['p95_ms']:16.4f} "
                    f"{custom_median_ms:16.4f} {custom_stats['p95_ms']:13.4f} "
                    f"{speedup:14.2f} "
                    f"{max_error:9.3e} {mean_error:10.3e}"
                )


if __name__ == "__main__":
    main()
