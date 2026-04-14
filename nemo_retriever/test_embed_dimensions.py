#!/usr/bin/env python3
"""Test: skip output shape resolution, allocate max buffer, and execute."""

import sys
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

NGC_MODELS = Path("/datasets/nv-ingest/models")
raw_path = NGC_MODELS / "embedding/ngc/hub/models--nim--nvidia--llama-nemotron-embed-1b-v2"

def find_engine(base: Path) -> Path:
    best, best_size = None, 0
    for f in base.rglob("*"):
        if f.is_file() and f.stat().st_size > best_size:
            best, best_size = f, f.stat().st_size
    return best

engine_path = find_engine(raw_path)
print(f"Engine: {engine_path}")
print(f"TRT:    {trt.__version__}")
print(f"Torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
runtime.engine_host_code_allowed = True

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

device = torch.device("cuda")
torch.cuda.init()

# ── Check for IOutputAllocator support ──────────────────────────────
has_output_allocator = hasattr(trt, "IOutputAllocator")
print(f"Has IOutputAllocator: {has_output_allocator}")

# ── Test matrix ─────────────────────────────────────────────────────
test_cases = [
    # (B, S, profile_idx, dim_val)
    (1,  64, 0, 2048),
    (1,  64, 0, 1024),
    (1,  64, 0, 384),
    (2,  64, 0, 2048),
    (2,  64, 0, 1024),
    (2,  64, 0, 384),
    (2,  64, 0, 0),
    (1, 128, 0, 2048),
]

MAX_DIM = 2048  # allocate output assuming this max

for B, S, profile_idx, dim_val in test_cases:
    tag = f"B={B} S={S} profile={profile_idx} dim_val={dim_val}"
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream(device)

    # Select profile
    ctx.set_optimization_profile_async(profile_idx, stream.cuda_stream)
    stream.synchronize()

    # Inputs
    d_input_ids      = torch.ones((B, S), dtype=torch.int64, device=device)
    d_attention_mask  = torch.ones((B, S), dtype=torch.int64, device=device)
    d_dims            = torch.full((B,), dim_val, dtype=torch.int64, device=device)

    ctx.set_input_shape("input_ids", (B, S))
    ctx.set_tensor_address("input_ids", d_input_ids.data_ptr())
    ctx.set_input_shape("attention_mask", (B, S))
    ctx.set_tensor_address("attention_mask", d_attention_mask.data_ptr())
    ctx.set_input_shape("dimensions", (B,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())

    # Allocate MAX output buffer — ignore what TRT thinks the shape is
    d_output = torch.zeros((B, MAX_DIM), dtype=torch.float32, device=device)
    ctx.set_tensor_address("embeddings", d_output.data_ptr())

    # Check reported shape (expect -1 for data-dependent dim)
    try:
        reported = tuple(ctx.get_tensor_shape("embeddings"))
    except Exception as e:
        reported = f"ERROR({e})"

    # Execute
    with torch.cuda.stream(stream):
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    if ok:
        # Figure out how many dims are actually populated
        row = d_output[0].cpu().numpy()
        # Find last non-zero index
        nonzero_idx = np.nonzero(np.abs(row) > 1e-12)[0]
        if len(nonzero_idx) > 0:
            actual_dim = int(nonzero_idx[-1]) + 1
        else:
            actual_dim = 0
        norm = torch.norm(d_output).item()
        total_nz = (d_output.abs() > 1e-12).sum().item()
        print(f"  {tag}  reported={reported}  exec=OK  "
              f"norm={norm:.4f}  nonzero={total_nz}/{d_output.numel()}  "
              f"actual_dim≈{actual_dim}  first_5={row[:5].tolist()}")
    else:
        print(f"  {tag}  reported={reported}  exec=FAILED")

    del ctx

print("\nDone.")
