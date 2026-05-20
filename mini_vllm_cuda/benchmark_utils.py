import torch


def benchmark_cuda(fn, warmup: int = 20, iters: int = 100) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _percentile(sorted_values, percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def benchmark_cuda_stats(
    fn,
    warmup: int = 50,
    repeats: int = 200,
    inner_iters: int = 20,
):
    """Return robust CUDA latency stats in milliseconds per call.

    Each repeat measures `inner_iters` calls inside one CUDA event interval, then
    stores the average per call for that sample. This reduces random noise for
    very small kernels while still letting us report median and p95 across many
    independent samples.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if inner_iters <= 0:
        raise ValueError("inner_iters must be positive")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(inner_iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner_iters)

    samples_sorted = sorted(samples)
    mean_ms = sum(samples) / len(samples)
    return {
        "mean_ms": mean_ms,
        "median_ms": _percentile(samples_sorted, 50.0),
        "p05_ms": _percentile(samples_sorted, 5.0),
        "p95_ms": _percentile(samples_sorted, 95.0),
        "min_ms": samples_sorted[0],
        "max_ms": samples_sorted[-1],
        "repeats": repeats,
        "inner_iters": inner_iters,
        "warmup": warmup,
    }
