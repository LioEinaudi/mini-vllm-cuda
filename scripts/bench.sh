#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
"${PYTHON:-python3}" benchmarks/bench_rmsnorm.py
"${PYTHON:-python3}" benchmarks/bench_rope.py
"${PYTHON:-python3}" benchmarks/bench_decode_attention.py
"${PYTHON:-python3}" benchmarks/bench_int8_gemv.py
