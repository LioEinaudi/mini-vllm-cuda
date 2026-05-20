import pytest
import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.torch_ref import int8_gemv_ref


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not mvc.is_extension_available()),
    reason="CUDA extension is not available",
)


def test_int8_gemv_matches_torch_ref():
    torch.manual_seed(0)
    x = torch.randn(256, device="cuda", dtype=torch.float16)
    w_int8 = torch.randint(-127, 128, (128, 256), device="cuda", dtype=torch.int8)
    scales = torch.rand(128, device="cuda", dtype=torch.float16) * 0.02

    actual = mvc.int8_gemv(x, w_int8, scales)
    expected = int8_gemv_ref(x, w_int8, scales)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
