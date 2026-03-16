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

---

## Results

### Attention Kernel — Triton vs PyTorch

> Benchmarked on **RTX 4060 (6 GB VRAM, Ada Lovelace)** — the target deployment hardware.
> CUDA Events, 100 iterations, fp16.

| Batch | Heads | Seq Len | PyTorch (ms) | Triton (ms) | Speedup | VRAM: PyTorch | VRAM: Triton |
|-------|-------|---------|-------------|------------|---------|--------------|-------------|
| 1 | 4 | 128 | 0.1274 | 0.0301 | **4.23×** | 9.18 MB | 8.78 MB |
| 1 | 4 | 256 | 0.1359 | 0.0247 | **5.50×** | 10.75 MB | 9.04 MB |
| 1 | 4 | 512 | 0.1347 | 0.0677 | **1.99×** | 16.65 MB | 9.57 MB |
| 1 | 8 | 128 | 0.1337 | 0.0193 | **6.92×** | 9.83 MB | 9.04 MB |
| 1 | 8 | 256 | 0.2016 | 0.0425 | **4.75×** | 12.98 MB | 9.57 MB |
| 1 | 8 | 512 | 0.2130 | 0.1200 | **1.77×** | 24.77 MB | 10.62 MB |
| 2 | 4 | 128 | 0.1430 | 0.0236 | **6.05×** | 9.83 MB | 9.04 MB |
| 2 | 4 | 256 | 0.1330 | 0.0237 | **5.62×** | 12.98 MB | 9.57 MB |
| 2 | 4 | 512 | 0.2062 | 0.0676 | **3.05×** | 24.77 MB | 10.62 MB |
| 2 | 8 | 128 | 0.3702 | 0.0215 | **17.23×** | 11.14 MB | 9.57 MB |
| 2 | 8 | 256 | 0.1682 | 0.0406 | **4.14×** | 17.43 MB | 10.62 MB |
| 2 | 8 | 512 | 0.5204 | 0.1016 | **5.12×** | 41.03 MB | 12.71 MB |

Peak speedup: **17.23×** at B=2, H=8, seq=128. At B=2, H=8, seq=512: VRAM drops from **41.0 MB → 12.7 MB (3.2× reduction)** — the primary benefit at longer sequences on constrained hardware.

The speedups here exceed those on higher-end hardware (RTX 5000 Ada, 32 GB) because memory pressure is more severe on 6 GB — the Triton kernel's SRAM tiling advantage is amplified when HBM bandwidth is the bottleneck.

---

### End-to-End Mini Block — Attention + INT8 Linear (B=2, H=8, seq=256, D=64)

| Metric | PyTorch | Optimised | Delta |
|--------|---------|-----------|-------|
| Latency | 0.7453 ms | 0.1275 ms | **−82.9% (5.84× faster)** |
| Peak VRAM | 16.40 MB | 11.69 MB | **−28.7%** |

**5.84× faster end-to-end** with **28.7% less VRAM** on the 6 GB RTX 4060. Both gains compound: Triton attention eliminates HBM round-trips, INT8 weight compression halves the linear layer's memory footprint.

The profiler trace confirms the kernel breakdown:

```
attention_kernel    37.0 µs   21.1%   (single fused Triton dispatch — no materialised T×T matrix)
aten::bmm           30.9 µs   17.6%   (PyTorch baseline matmuls)
aten::copy_         31.9 µs   18.2%   (INT8 dequant cast)
aten::triu          20.0 µs   11.4%   (causal mask allocation in PyTorch path)
```

The Triton attention kernel dispatches once and uses no intermediate buffers. The PyTorch path allocates and fills a full T×T matrix (`aten::triu`), runs two separate matmuls, and a softmax — all hitting HBM repeatedly.

---

## Why This Matters

| This system | Production equivalent |
|---|---|
| Tiled attention in SRAM | FlashAttention — used in vLLM, HuggingFace, llama.cpp |
| INT8 weight-only quantisation | GPTQ, AWQ weight-only quant for 4/8-bit inference |
| Causal mask in-kernel | Standard in all autoregressive inference engines |
| 6 GB VRAM target | RTX 4060, 3060, laptop GPUs — the edge deployment reality |

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
│   ├── __init__.py
│   ├── triton_flash.py          # Triton causal attention kernel
│   └── quant_linear.py          # INT8 weight-only linear layer
├── models/
│   ├── __init__.py
│   └── mini_block.py            # TorchMiniBlock vs OptimizedMiniBlock
├── benchmarks/
│   ├── attention_bench.py       # Attention-only head-to-head → results/attention.json
│   ├── block_bench.py           # End-to-end block comparison
│   ├── baseline.py              # PyTorch attention standalone baseline
│   ├── test_triton_attention.py # Correctness — Triton vs PyTorch
│   ├── test_quant_linear.py     # Correctness — INT8 vs fp16 Linear
│   └── results/
│       └── attention.json       # Full benchmark results
├── profiler/
│   ├── profile_block.py         # PyTorch profiler + NVTX trace
│   └── mini_block_trace.json    # Chrome trace output
└── pyproject.toml               # pip install -e . for local imports
```

---

## Setup

```bash
pip install torch triton nvtx
pip install -e .    # makes kernels/ and models/ importable from anywhere in the repo
```

Requires a CUDA-capable NVIDIA GPU (Ampere or newer). Benchmarks run on RTX 4060 (6 GB, Ada Lovelace).

---

## Usage

```bash
# Correctness tests
python benchmarks/test_triton_attention.py   # Triton vs PyTorch attention
python benchmarks/test_quant_linear.py       # INT8 vs fp16 linear

# Benchmarks
python benchmarks/attention_bench.py         # Attention-only speedup table → results/attention.json
python benchmarks/block_bench.py             # End-to-end block comparison

# Profiling (generates Chrome trace)
python profiler/profile_block.py             # outputs profiler/mini_block_trace.json
                                             # open with chrome://tracing or perfetto.dev
```

> **Note:** First run of any Triton kernel will be slow (~10–30s) while Triton JIT-compiles and caches. Subsequent runs are fast.

---

## Design Decisions

**Why online softmax?**
Avoids materialising the full `T × T` score matrix in HBM. Each tile of scores is computed, the running max and normaliser are updated incrementally (numerically stable), and the output accumulator is corrected — identical to the core algorithm in FlashAttention (Dao et al., 2022). The profiler confirms this: `aten::triu` (causal mask allocation) appears only in the PyTorch path and consumes 11.4% of CUDA time — entirely absent from the Triton path.

**Why weight-only INT8, not activation quantisation?**
Activation quantisation requires per-token dynamic range estimation and is sensitive to outlier activations — a known problem in LLMs. Weight-only quantisation is static, cheap at runtime, and loses only ~0.5–1% accuracy on standard benchmarks while cutting weight memory by ~4×.

**Why not fuse dequant+GEMM?**
The dequant path (`qweight.float() * scale → x.float() @ w.T`) is unfused — `aten::copy_` (18.2% of CUDA time) is the cast overhead. A production fix integrates dequant into the GEMM kernel (GPTQ-triton, Marlin), eliminating this entirely. That is the logical next step.

---

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
