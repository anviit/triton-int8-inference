# GPU-Efficient LLM Inference Engine (v0.1)

## Problem
LLM inference on mid-tier GPUs is memory-bound and inefficient.
Standard PyTorch attention materializes large intermediate tensors,
and fp16 linear layers consume significant VRAM, limiting throughput
and batch size on 6 GB GPUs.

## Constraints
- Single GPU (6 GB VRAM)
- Inference only (no backward)
- fp16 activations
- Causal self-attention

## Optimizations
### 1. Triton Causal Attention
- Forward-only attention kernel
- Blocked QKᵀ + softmax + V
- No attention matrix materialization
- Causal masking in-kernel

### 2. Weight-Only INT8 Linear
- Per-output-channel INT8 quantization
- fp16 activations
- On-the-fly dequantization
- ~4× weight memory reduction

## Results

### Attention Only
| Seq Len | Speedup | VRAM Reduction |
|--------|---------|----------------|
| 256    | ~8×     | ~2×            |
| 512    | ~3×     | ~3×            |

### End-to-End Mini Block
| Metric    | PyTorch  | Optimized |
|-----------|----------|-----------|
| Latency   |0.1613 ms | 0.1613 ms |
| Peak VRAM | 16.4 MB  | 11.7 MB   |

## Profiling Insights
- Triton attention significantly reduces attention latency and memory
- Quantized linear layers reduce VRAM but introduce dequantization overhead
- End-to-end latency tradeoff is dominated by INT8 dequant cost
- Memory savings enable higher batch sizes and prevent OOM on constrained GPUs

## Tradeoffs
- Inference-only (no backward)
- Fixed head dimension
- INT8 linear layers trade latency for memory
- No kernel fusion between dequantization and GEMM

## Why This Matters
This engine demonstrates how targeted kernel-level optimizations
can significantly improve LLM inference under real hardware constraints,
without retraining or model changes.



## Model-Level Validation

This inference engine is validated on a real transformer model.

▶️ **GPT-2 Triton Inference Demo**  
https://github.com/dunkinflicka/triton_gpt2

The demo integrates the Triton causal attention kernel from this repository
into a GPT-2 implementation (nanoGPT) and benchmarks end-to-end
autoregressive inference throughput and memory usage.
