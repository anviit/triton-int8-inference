"""
PyTorch profiler + NVTX trace for TorchMiniBlock vs OptimizedMiniBlock.
Outputs a Chrome trace to profiler/mini_block_trace.json.
Open with chrome://tracing or perfetto.dev.
"""

import torch
import nvtx
from torch.profiler import profile, record_function, ProfilerActivity

from models.mini_block import TorchMiniBlock, OptimizedMiniBlock


def run(model: torch.nn.Module, x: torch.Tensor, tag: str) -> None:
    with nvtx.annotate(tag):
        model(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 2, 8, 256, 64
    x = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    torch_block = TorchMiniBlock(D).cuda().half()
    opt_block = OptimizedMiniBlock(D)

    # Warmup — outside the profiler window
    for _ in range(10):
        run(torch_block, x, "torch_warmup")
        run(opt_block, x, "opt_warmup")

    # Profile both blocks in one trace
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        torch.cuda.synchronize()
        run(torch_block, x, "torch_block")
        torch.cuda.synchronize()
        run(opt_block, x, "optimized_block")
        torch.cuda.synchronize()

    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    )

    # Export Chrome trace — must be inside or after the `with` block
    prof.export_chrome_trace("profiler/mini_block_trace.json")
    print("\nTrace saved → profiler/mini_block_trace.json")
    print("Open with chrome://tracing or https://ui.perfetto.dev")
