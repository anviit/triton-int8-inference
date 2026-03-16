# kernel-triton

GPU-efficient LLM inference on constrained hardware — custom Triton causal attention kernel with INT8 weight-only quantization. Targets 6 GB VRAM mid-tier GPUs.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Triton](https://img.shields.io/badge/OpenAI-Triton-412991?style=flat-square)](https://github.com/openai/triton)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Ampere+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

---

## Problem

Standard PyTorch attention materialises the full `T × T` score matrix in HBM — an O(T²) memory footprint that constrains batch size and sequence length on 6 GB GPUs. fp16 linear layers add further memory pressure at inference time, where the backward pass is never needed.

**Constraints:** single GPU (6 GB VRAM) · inference-only (no backward) · fp16 activations · causal self-attention.

---

## Optimisations

### 1. Triton Causal Attention Kernel

Blocked `QKᵀ` + online softmax + `V` accumulation in SRAM — no intermediate attention matrix written to HBM.

- Online softmax (numerically stable, single-pass)
- Causal mask applied in-kernel with no separate allocation
- Fused into one kernel dispatch — no intermediate tensors between Q@K, softmax, and @V
- Head dimension fixed at 64; block sizes tunable per hardware

### 2. Weight-Only INT8 Linear

Weights quantised to INT8 per output channel at load time, dequantised on the fly during forward pass.

- ~4× weight memory reduction vs fp16 baseline
- fp16 activations preserved — no activation quantisation error
- Per-output-channel scaling: `scale = max(|W|, axis=1) / 127`
- Tradeoff: dequantisation overhead is real; latency is neutral end-to-end, but VRAM savings enable larger batch sizes on constrained hardware

---

## Results

### Attention Kernel — Triton vs PyTorch (CUDA Events, 100 iterations, fp16)

> Benchmarked on a local NVIDIA GPU. All comparisons use fp16.

| Batch | Heads | Seq Len | PyTorch (ms) | Triton (ms) | Speedup | VRAM: PyTorch | VRAM: Triton |
|-------|-------|---------|-------------|------------|---------|--------------|-------------|
| 1 | 4 | 128 | 0.4885 | 0.0329 | **14.87×** | 9.18 MB | 8.78 MB |
| 1 | 4 | 256 | 0.8439 | 0.1022 | **8.26×** | 10.75 MB | 9.04 MB |
| 1 | 4 | 512 | 0.2589 | 0.0695 | **3.72×** | 16.65 MB | 9.57 MB |
| 1 | 8 | 128 | 0.2616 | 0.0600 | **4.36×** | 9.83 MB | 9.04 MB |
| 1 | 8 | 256 | 0.1524 | 0.0500 | **3.05×** | 12.98 MB | 9.57 MB |
| 1 | 8 | 512 | 0.2380 | 0.1213 | **1.96×** | 24.77 MB | 10.62 MB |
| 2 | 4 | 128 | 0.2490 | 0.0539 | **4.62×** | 9.83 MB | 9.04 MB |
| 2 | 4 | 256 | 0.2724 | 0.0813 | **3.35×** | 12.98 MB | 9.57 MB |
| 2 | 4 | 512 | 0.1959 | 0.0686 | **2.86×** | 24.77 MB | 10.62 MB |
| 2 | 8 | 128 | 0.1541 | 0.0276 | **5.59×** | 11.14 MB | 9.57 MB |
| 2 | 8 | 256 | 0.1606 | 0.0374 | **4.29×** | 17.43 MB | 10.62 MB |
| 2 | 8 | 512 | 0.4268 | 0.1483 | **2.88×** | 41.03 MB | 12.71 MB |

Speedup is most pronounced at small sequence lengths where SRAM tiling gives the largest relative advantage over HBM round-trips. At B=2, H=8, seq=512: VRAM drops from **41.0 MB → 12.7 MB (3.2× reduction)** — the primary benefit at longer sequences where memory pressure dominates.

---

### End-to-End Mini Block — Attention + INT8 Linear (B=2, H=8, seq=256, D=64)

| Metric | PyTorch | Optimised | Delta |
|--------|---------|-----------|-------|
| Latency | 0.1613 ms | 0.1617 ms | ~neutral |
| Peak VRAM | 16.4 MB | 11.7 MB | **−28.7%** |

**Why latency is neutral end-to-end:** the Triton attention kernel is significantly faster in isolation, but the INT8 dequantisation path (`qweight.float() * scale`, then `x.float() @ w.T`) introduces a type-cast and elementwise multiply that absorbs the attention savings. This is a deliberate and documented tradeoff — the goal on a 6 GB GPU is fitting more work into memory, not shaving single-request latency.

In practice, 28.7% VRAM reduction at the block level means running batch size 4 instead of 3 within the same memory budget — a real system-level throughput improvement even with neutral single-request latency.

A production-grade fix would fuse the dequant into the GEMM via a custom kernel (as in GPTQ-triton and Marlin). That is the logical next step.

---

## Why This Matters

| This system | Production equivalent |
|---|---|
| Tiled attention in SRAM | FlashAttention — used in vLLM, HuggingFace, llama.cpp |
| INT8 weight-only quantisation | GPTQ, AWQ weight-only quant for 4/8-bit inference |
| Causal mask in-kernel | Standard in all autoregressive inference engines |
| VRAM-constrained batching | Edge deployment, consumer GPU serving |

---

## Model-Level Validation

The Triton attention kernel is validated on a real transformer end-to-end.

▶️ **GPT-2 Triton Inference Demo** — [`dunkinflicka/triton_gpt2`](https://github.com/dunkinflicka/triton_gpt2)

Integrates this kernel into a nanoGPT implementation and benchmarks autoregressive inference throughput and memory usage against the PyTorch baseline.

---

## Correctness

```
✅ Triton attention matches PyTorch   rtol=1e-2, atol=1e-2, fp16
✅ QuantLinear output matches fp16 Linear   rtol=2e-1, atol=2e-1 (expected for INT8)
```

---

## Project Structure

```
kernel-triton/
├── kernels/
│   ├── triton_flash.py          # Triton causal attention kernel
│   └── quant_linear.py          # INT8 weight-only linear layer
├── models/
│   └── mini_block.py            # TorchMiniBlock vs OptimizedMiniBlock
├── benchmarks/
│   ├── attention_bench.py       # Attention-only head-to-head
│   ├── block_bench.py           # End-to-end block comparison
│   ├── baseline.py              # PyTorch attention standalone baseline
│   ├── test_triton_attention.py # Correctness — Triton vs PyTorch
│   ├── test_quant_linear.py     # Correctness — INT8 vs fp16 Linear
│   └── results/
│       └── attention.json       # Full benchmark results
└── profiler/
    ├── profile_block.py         # PyTorch profiler + NVTX trace
    └── mini_block_trace.json    # Chrome trace output
```

---

## Setup

```bash
pip install torch triton nvtx
```

Requires a CUDA-capable NVIDIA GPU (Ampere or newer recommended).

---

## Usage

```bash
# Correctness tests
python benchmarks/test_triton_attention.py   # Triton vs PyTorch attention
python benchmarks/test_quant_linear.py       # INT8 vs fp16 linear

# Benchmarks
python benchmarks/attention_bench.py         # Attention-only speedup table → results/attention.json
python benchmarks/block_bench.py             # End-to-end block comparison

# Profiling
python profiler/profile_block.py             # Outputs profiler/mini_block_trace.json
                                             # Open in chrome://tracing
```

> **Note:** First run of any Triton kernel will be slow (~10–30s) while Triton JIT-compiles and caches. Subsequent runs are fast.

---

## Design Decisions

**Why online softmax?**
Avoids materialising the full `T × T` score matrix in HBM. Each tile of scores is computed, the running max and normaliser are updated incrementally (numerically stable), and the output accumulator is corrected — identical to the core algorithm in FlashAttention (Dao et al., 2022).

**Why weight-only INT8, not activation quantisation?**
Activation quantisation requires per-token dynamic range estimation and is sensitive to outlier activations — a known problem in LLMs. Weight-only quantisation is static, cheap at runtime, and loses only ~0.5–1% accuracy on standard benchmarks while cutting weight memory by ~4×.

**Why is end-to-end latency neutral?**
The dequant path is not fused with the GEMM. A production fix fuses dequant into the matrix multiply kernel (GPTQ-triton, Marlin). The value here is VRAM reduction, not latency.

---

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
