# mini-vllm-cuda

An educational C++/CUDA + PyTorch extension project for LLM decode-stage inference kernels.

The goal is not to train models and not to build a chat app. This project focuses on the core CUDA operators used during autoregressive decode:

- RMSNorm
- RoPE
- Decode attention with a continuous KV cache
- A future simplified paged KV cache
- INT8 weight-only GEMV
- Correctness tests, latency benchmarks, and future Nsight profiling scripts

The initial scaffold is intentionally small. RMSNorm now has a hand-written CUDA v0 implementation with correctness tests and latency benchmarks. RoPE, decode attention, and INT8 GEMV are still correctness placeholders or scaffolds that will be replaced one operator at a time.

## Current Status

- RMSNorm v0 implemented as a hand-written CUDA kernel.
- RMSNorm correctness tested for float32 and float16.
- RMSNorm latency benchmark reports median and p95 timings.
- RMSNorm v0 performance analysis has been recorded in the local learning notes.
- RoPE, decode attention, and INT8 GEMV are still placeholder/scaffold implementations.
- Next step: RMSNorm v1 design or RoPE v0, after the RMSNorm v0 baseline is reviewed.

## Project Layout

```text
mini-vllm-cuda/
├── README.md
├── setup.py
├── pyproject.toml
├── csrc/
│   ├── bindings.cpp
│   ├── rmsnorm.cu
│   ├── rope.cu
│   ├── decode_attention.cu
│   ├── int8_gemv.cu
│   └── common.h
├── mini_vllm_cuda/
│   ├── __init__.py
│   ├── ops.py
│   ├── torch_ref.py
│   └── benchmark_utils.py
├── tests/
│   ├── test_rmsnorm.py
│   ├── test_rope.py
│   ├── test_decode_attention.py
│   └── test_int8_gemv.py
├── benchmarks/
│   ├── bench_rmsnorm.py
│   ├── bench_rope.py
│   ├── bench_decode_attention.py
│   └── bench_int8_gemv.py
└── scripts/
    ├── build.sh
    ├── test.sh
    └── bench.sh
```

## Requirements

- Linux
- Python 3.9+
- PyTorch with CUDA support
- CUDA Toolkit compatible with your PyTorch build
- NVIDIA GPU. The default architecture is `sm_89`, suitable for RTX 4060 Laptop.

To override the architecture:

```bash
export MVC_CUDA_ARCH_LIST="8.6"
pip install -e . --no-build-isolation
```

You can also use PyTorch's native variable:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
pip install -e . --no-build-isolation
```

## Build Notes

Real setup and troubleshooting notes are recorded in [`docs/build_notes.md`](docs/build_notes.md).

## Install

From the project root:

```bash
pip install -e . --no-build-isolation
```

If CUDA or the extension build is unavailable, the Python package can still import, but calling CUDA ops will raise a clear error telling you to rebuild the extension.

## Python API

```python
import mini_vllm_cuda as mvc

y = mvc.rmsnorm(x, weight, eps)
q_out, k_out = mvc.rope(q, k, cos, sin)
out = mvc.decode_attention(q, k_cache, v_cache, seq_len)
y = mvc.int8_gemv(x, w_int8, scales)
```

## Test

```bash
pytest tests/
```

Each test compares the extension output against a PyTorch reference implementation in `mini_vllm_cuda/torch_ref.py`.

## Benchmark

```bash
python benchmarks/bench_rmsnorm.py
```

Or run all benchmarks:

```bash
bash scripts/bench.sh
```

All benchmarks use `torch.cuda.Event` timing.

## Development Plan

1. Use the RMSNorm v0 baseline to design and validate a v1 optimization.
2. Replace `csrc/rope.cu` with a hand-written CUDA RoPE v0 kernel.
3. Implement decode attention over a continuous KV cache with owner-authored online softmax.
4. Add a simplified paged KV cache and block table.
5. Implement INT8 weight-only GEMV with per-channel scales.
6. Add Nsight Compute / Nsight Systems profiling scripts.

The test and benchmark files are already in place so each kernel can be implemented and optimized without changing the public API.
