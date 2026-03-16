"""
Transformer mini block — baseline vs optimised inference.

TorchMiniBlock:     standard PyTorch attention + fp16 Linear
OptimizedMiniBlock: Triton causal attention + INT8 weight-only Linear
"""

import torch
import torch.nn as nn

from kernels.triton_flash import triton_attention
from kernels.quant_linear import QuantLinear


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive causal self-attention — materialises the full T×T score matrix in HBM."""
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d ** 0.5
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class TorchMiniBlock(nn.Module):
    """Baseline: standard PyTorch attention + fp16 Linear."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, M, D]
        attn = torch_attention(x, x, x)
        return self.linear(attn)


class OptimizedMiniBlock(nn.Module):
    """Optimised: Triton causal attention + INT8 weight-only Linear."""

    def __init__(self, dim: int):
        super().__init__()
        weight = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
        self.linear = QuantLinear(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, M, D]
        attn = triton_attention(x, x, x)
        return self.linear(attn)
