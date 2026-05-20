import math

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps) * weight.float()
    return y.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_ref(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def decode_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    k = k_cache[:seq_len].float()
    v = v_cache[:seq_len].float()
    q_float = q.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("hd,shd->hs", q_float, k) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,shd->hd", attn, v)
    return out.to(dtype=q.dtype)


def int8_gemv_ref(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = scales.float()
    if scale.dim() == 1:
        scale = scale[:, None]
    w = w_int8.float() * scale
    return torch.matmul(w, x.float())
