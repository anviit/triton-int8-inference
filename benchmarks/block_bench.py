"""
End-to-end mini block benchmark: TorchMiniBlock vs OptimizedMiniBlock.

Result (B=2, H=8, seq=256, D=64):
  Latency: 0.2026 ms → 0.0958 ms  (-52.7%, 2.1× faster)
  VRAM:    16.40 MB  → 11.69 MB   (-28.7%)

Both Triton attention and INT8 weight compression contribute.
The dequant cast (aten::copy_) is visible in the profiler trace but
does not eliminate the attention gain at this config.
"""

import torch
from models.mini_block import TorchMiniBlock, OptimizedMiniBlock


ITERS = 100


def benchmark(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    # Warmup
    for _ in range(10):
        model(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        model(x)
    end.record()

    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / ITERS
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    return latency_ms, peak_vram_mb


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 2, 8, 256, 64
    x = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    torch_block = TorchMiniBlock(D).cuda().half()
    opt_block = OptimizedMiniBlock(D)

    torch_lat, torch_mem = benchmark(torch_block, x)
    opt_lat, opt_mem = benchmark(opt_block, x)

    latency_delta_pct = (opt_lat - torch_lat) / torch_lat * 100
    vram_delta_pct = (torch_mem - opt_mem) / torch_mem * 100
    speedup = torch_lat / opt_lat

    print("\n=== MINI BLOCK COMPARISON ===")
    print(f"  Config     : B={B}, H={H}, M={M}, D={D}")
    print(f"  PyTorch    : {torch_lat:.4f} ms | {torch_mem:.2f} MB")
    print(f"  Optimised  : {opt_lat:.4f} ms | {opt_mem:.2f} MB")
    print(f"  Latency Δ  : {latency_delta_pct:+.1f}% ({speedup:.2f}× faster)")
    print(f"  VRAM Δ     : -{vram_delta_pct:.1f}% ({torch_mem:.1f} MB → {opt_mem:.1f} MB)")
