"""
Weight-only INT8 quantised Linear layer.
Inference-only — no backward pass.

Quantisation scheme: per-output-channel symmetric INT8.
  scale = max(|W|, axis=1) / 127
  qweight = round(W / scale)

Forward: dequantise weights on the fly, then matmul with fp16 activations.
Tradeoff: dequant is not fused with GEMM — latency is neutral vs fp16 baseline,
but weight memory is reduced ~4×. A fused dequant+GEMM kernel (Marlin, GPTQ-triton)
would recover latency at the cost of implementation complexity.
"""

import torch
import torch.nn as nn


class QuantLinear(nn.Module):

    def __init__(self, weight_fp16: torch.Tensor):
        super().__init__()

        assert weight_fp16.ndim == 2, "Expected [out_features, in_features]"
        assert weight_fp16.dtype == torch.float16

        # Per-output-channel scale: max absolute value per row
        max_val = weight_fp16.abs().max(dim=1, keepdim=True)[0]
        scale = (max_val / 127.0).clamp(min=1e-6)

        # Quantise to INT8
        qweight = torch.round(weight_fp16 / scale).to(torch.int8)

        # Register as buffers (not trainable parameters)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]  (fp16)
        returns: [..., out_features]  (fp16)
        """
        # Dequantise: INT8 → fp32, then scale
        w = self.qweight.float() * self.scale  # [out, in]
        out = x.float() @ w.T                  # [..., out]
        return out.to(x.dtype)
