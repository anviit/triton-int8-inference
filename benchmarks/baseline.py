"""
Standalone PyTorch attention baseline.
Benchmarks naive causal attention on a single config and saves results to JSON.
"""

import json
import torch
from pathlib import Path


BATCH = 1
HEADS = 4
SEQ_LEN = 256
DIM = 64
DEVICE = "cuda"
ITERS = 100


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive causal self-attention — materialises the full T×T score matrix."""
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d ** 0.5
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def benchmark() -> dict:
    torch.manual_seed(0)

    # Shape: [B, H, SEQ_LEN, DIM] — matches attention_bench.py and triton kernel
    q = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(10):
        torch_attention(q, k, v)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        torch_attention(q, k, v)
    end.record()

    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / ITERS
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "batch": BATCH,
        "heads": HEADS,
        "seq_len": SEQ_LEN,
        "dim": DIM,
        "latency_ms": round(latency_ms, 6),
        "peak_vram_mb": round(peak_vram_mb, 3),
    }


if __name__ == "__main__":
    results = benchmark()

    print("\nBASELINE RESULTS")
    print("-" * 40)
    for key, val in results.items():
        print(f"  {key}: {val}")

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {out_dir / 'baseline.json'}")
