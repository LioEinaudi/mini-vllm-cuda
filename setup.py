import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except Exception as exc:  # pragma: no cover - setup-time fallback
    torch = None
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ROOT = Path(__file__).parent.resolve()


def make_extensions():
    if torch is None or CUDAExtension is None or CUDA_HOME is None:
        reason = (
            f"torch import failed: {_TORCH_IMPORT_ERROR}"
            if _TORCH_IMPORT_ERROR is not None
            else "CUDA_HOME was not found"
        )
        print(f"[mini-vllm-cuda] CUDA extension will not be built: {reason}")
        return []

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("MVC_CUDA_ARCH_LIST", "8.9"))

    sources = [
        "csrc/bindings.cpp",
        "csrc/placeholders.cpp",
        "csrc/rmsnorm.cu",
    ]

    return [
        CUDAExtension(
            name="mini_vllm_cuda._C",
            sources=[str(ROOT / src) for src in sources],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ]


setup(
    name="mini-vllm-cuda",
    version="0.1.0",
    description="Educational CUDA kernels for LLM decode-stage inference.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch", "pytest"],
    ext_modules=make_extensions(),
    cmdclass={"build_ext": BuildExtension} if BuildExtension is not None else {},
)
