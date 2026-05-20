import pytest
import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.torch_ref import rope_ref


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not mvc.is_extension_available()),
    reason="CUDA extension is not available",
)


def test_rope_matches_torch_ref():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)
    cos = torch.randn(2, 1, 64, device="cuda", dtype=torch.float16)
    sin = torch.randn(2, 1, 64, device="cuda", dtype=torch.float16)

    actual_q, actual_k = mvc.rope(q, k, cos, sin)
    expected_q, expected_k = rope_ref(q, k, cos, sin)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-3)
