import importlib

import torch  # noqa: F401 - load PyTorch shared libraries before importing _C

try:
    _C = importlib.import_module("mini_vllm_cuda._C")
except Exception as exc:  # pragma: no cover - depends on local build state
    _C = None
    _EXTENSION_IMPORT_ERROR = exc
else:
    _EXTENSION_IMPORT_ERROR = None


def is_extension_available() -> bool:
    return _C is not None


def extension_error():
    return _EXTENSION_IMPORT_ERROR


def _require_extension():
    if _C is None:
        raise RuntimeError(
            "mini_vllm_cuda CUDA extension is not available. "
            "Build it with `pip install -e .` from the mini-vllm-cuda project root. "
            f"Original import error: {_EXTENSION_IMPORT_ERROR!r}"
        )
    return _C


def rmsnorm(x, weight, eps=1e-6):
    return _require_extension().rmsnorm(x, weight, float(eps))


def rmsnorm_v1(x, weight, eps=1e-6):
    return _require_extension().rmsnorm_v1(x, weight, float(eps))


def rope(q, k, cos, sin):
    return tuple(_require_extension().rope(q, k, cos, sin))


def decode_attention(q, k_cache, v_cache, seq_len):
    return _require_extension().decode_attention(q, k_cache, v_cache, int(seq_len))


def int8_gemv(x, w_int8, scales):
    return _require_extension().int8_gemv(x, w_int8, scales)
