import pytest
import torch

import mini_vllm_cuda as mvc
from mini_vllm_cuda.torch_ref import decode_attention_ref


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not mvc.is_extension_available()),
    reason="CUDA extension is not available",
)


def test_decode_attention_matches_torch_ref():
    torch.manual_seed(0)
    seq_len = 96
    q = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(128, 8, 64, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(128, 8, 64, device="cuda", dtype=torch.float16)

    actual = mvc.decode_attention(q, k_cache, v_cache, seq_len)
    expected = decode_attention_ref(q, k_cache, v_cache, seq_len)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
