# mini-vllm-cuda

`mini-vllm-cuda` is a C++/CUDA + PyTorch extension project for studying and implementing CUDA kernels used in LLM decode-stage inference. The repository emphasizes a small, inspectable workflow: write a kernel, compare it against a PyTorch reference, run correctness tests, and measure latency with CUDA events.

The project currently focuses on operator-level implementations rather than a complete inference runtime.

## Implemented So Far

- RMSNorm v0 CUDA kernel in `csrc/rmsnorm.cu`.
- PyTorch reference implementations in `mini_vllm_cuda/torch_ref.py`.
- PyTorch C++/CUDA extension binding through `csrc/bindings.cpp`.
- dtype dispatch for RMSNorm float32 and float16 inputs.
- pytest correctness tests for RMSNorm float32 and float16.
- RMSNorm latency benchmark with median and p95 timing.
- Build and environment notes in `docs/build_notes.md`.
- RMSNorm benchmark methodology and interpretation are maintained in local learning notes; public Nsight profiling notes are planned.

RoPE, decode attention, and INT8 GEMV currently remain placeholder/scaffold implementations.

## RMSNorm v0

RMSNorm v0 is the first hand-written CUDA kernel in this repository.

Input and output:

```text
x:      [num_tokens, hidden_size]
weight: [hidden_size]
y:      [num_tokens, hidden_size]
```

Formula:

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

Kernel mapping:

- one CUDA block handles one token row
- each thread processes columns with a strided loop
- each thread accumulates a partial `sum(x^2)` in fp32
- dynamic shared memory performs a tree reduction to compute the row sum
- each thread makes a second pass over the row and writes normalized output

Current limitations:

- no warp-level reduction yet
- no vectorized load/store path yet
- no `half2` path yet
- block size is fixed at 256 threads
- input `x` is read twice in the v0 implementation

## Correctness and Benchmarking

Correctness is checked against PyTorch reference implementations. RMSNorm tests report numerical agreement for both float32 and float16.

Benchmarks use `torch.cuda.Event` timing. The RMSNorm benchmark reports median and p95 latency to make short kernel measurements less sensitive to occasional timing noise.

Run correctness tests:

```bash
pytest tests/
```

Run the RMSNorm benchmark:

```bash
python benchmarks/bench_rmsnorm.py
```

The benchmark output includes:

- PyTorch reference median latency
- PyTorch reference p95 latency
- custom CUDA median latency
- custom CUDA p95 latency
- median speedup
- max absolute error
- mean absolute error

Performance results depend on GPU model, PyTorch/CUDA versions, tensor shapes, and benchmark settings.

## Requirements

- Linux
- Python 3.9+
- PyTorch with CUDA support
- CUDA Toolkit compatible with the installed PyTorch build
- NVIDIA GPU

The default CUDA architecture is `sm_89`, suitable for RTX 4060 Laptop.

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

## Install

From the project root:

```bash
pip install -e . --no-build-isolation
```

If CUDA or the extension build is unavailable, the Python package can still import, but calling CUDA ops will raise a clear error telling you to rebuild the extension.

## Test

```bash
pytest tests/
```

Each test compares extension output against a PyTorch reference implementation in `mini_vllm_cuda/torch_ref.py`.

## Benchmark

```bash
python benchmarks/bench_rmsnorm.py
```

Or run all benchmark scripts:

```bash
bash scripts/bench.sh
```

## Python API

```python
import mini_vllm_cuda as mvc

y = mvc.rmsnorm(x, weight, eps)
q_out, k_out = mvc.rope(q, k, cos, sin)
out = mvc.decode_attention(q, k_cache, v_cache, seq_len)
y = mvc.int8_gemv(x, w_int8, scales)
```

At the current stage, only RMSNorm has a real custom CUDA implementation. The other APIs are kept in place as scaffolds while their CUDA kernels are developed.

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

## Roadmap

Planned kernels and tools:

- RMSNorm v1 optimization based on the v0 baseline
- RoPE CUDA kernel
- decode attention with continuous KV cache
- simplified paged KV cache
- INT8 weight-only GEMV
- Nsight Compute / Nsight Systems profiling scripts

These items are planned or scaffolded unless otherwise stated above.

## Build Notes

Setup and troubleshooting notes are recorded in:

```text
docs/build_notes.md
```

## Limitations

- This is not a full LLM inference engine.
- This project is not affiliated with vLLM.
- Currently only RMSNorm has a real custom CUDA implementation.
- RoPE, decode attention, and INT8 GEMV are still placeholders or scaffolds.
- The simplified paged KV cache is future work.
- Performance numbers depend on GPU, PyTorch/CUDA version, tensor shapes, and benchmark configuration.
