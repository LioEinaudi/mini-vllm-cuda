import pytest
import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.torch_ref import rmsnorm_ref


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not mvc.is_extension_available()),
    reason="CUDA extension is not available",
)


@pytest.mark.parametrize("hidden_size", [1024, 2048, 4096])
@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize(
    ("impl_name", "impl_fn"),
    [
        ("v0", mvc.rmsnorm),
        ("v1", mvc.rmsnorm_v1),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 2e-3, 2e-3),
    ],
)
def test_rmsnorm_matches_torch_ref(hidden_size, num_tokens, impl_name, impl_fn, dtype, rtol, atol):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    eps = 1e-6

    actual = impl_fn(x, weight, eps)
    expected = rmsnorm_ref(x, weight, eps)
    error = (actual.float() - expected.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    print(
        f"RMSNorm impl={impl_name} dtype={dtype} num_tokens={num_tokens} hidden_size={hidden_size} "
        f"max_error={max_error:.6g} mean_error={mean_error:.6g} allclose={is_close}"
    )
    assert is_close, (
        f"RMSNorm mismatch: impl={impl_name}, dtype={dtype}, num_tokens={num_tokens}, "
        f"hidden_size={hidden_size}, max_error={max_error:.6g}, "
        f"mean_error={mean_error:.6g}, rtol={rtol}, atol={atol}"
    )
