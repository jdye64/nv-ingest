#!/usr/bin/env python3
"""Test: verify embed engine works across ALL profiles and long sequences."""

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

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
runtime.engine_host_code_allowed = True

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

device = torch.device("cuda")
torch.cuda.init()

MAX_DIM = 2048

# Profile ranges (from inspect output):
#   profile[0]: B=1..32  S=2..128
#   profile[1]: B=1..32  S=2..256
#   profile[2]: B=1..16  S=2..512
#   profile[3]: B=1..2   S=2..4096
#   profile[4]: B=1..1   S=2..8192

test_cases = [
    # (B, S, profile, dim_val, description)
    (1,   64, 0, 2048, "short seq, profile 0"),
    (2,   64, 0, 2048, "batch 2, profile 0"),
    (32, 128, 0, 2048, "max profile 0"),
    (16, 256, 1, 2048, "profile 1"),
    (16, 512, 2, 2048, "profile 2 max"),
    (2,  770, 3, 2048, "FAILING CASE: profile 3, S=770"),
    (2,  850, 3, 2048, "FAILING CASE: profile 3, S=850"),
    (2, 1209, 3, 2048, "FAILING CASE: profile 3, S=1209"),
    (2, 4096, 3, 2048, "profile 3 max seq"),
    (1, 8192, 4, 2048, "profile 4 max seq"),
    # Also test without calling infer_shapes
]

print(f"\n{'='*80}")
print("TEST 1: Fresh context per case, NO infer_shapes()")
print(f"{'='*80}")

for B, S, prof, dim_val, desc in test_cases:
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream(device)

    ctx.set_optimization_profile_async(prof, stream.cuda_stream)
    stream.synchronize()

    d_input_ids     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_attn_mask     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_dims          = torch.full((B,), dim_val, dtype=torch.int64, device=device)
    d_output        = torch.zeros((B, MAX_DIM), dtype=torch.float32, device=device)

    ctx.set_input_shape("input_ids", (B, S))
    ctx.set_tensor_address("input_ids", d_input_ids.data_ptr())
    ctx.set_input_shape("attention_mask", (B, S))
    ctx.set_tensor_address("attention_mask", d_attn_mask.data_ptr())
    ctx.set_input_shape("dimensions", (B,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())
    ctx.set_tensor_address("embeddings", d_output.data_ptr())

    with torch.cuda.stream(stream):
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    if ok:
        norm = torch.norm(d_output).item()
        nz = (d_output.abs() > 1e-12).sum().item()
        print(f"  OK   B={B:2d} S={S:5d} p={prof} dim={dim_val}  "
              f"norm={norm:.4f} nz={nz}/{B*MAX_DIM}  [{desc}]")
    else:
        print(f"  FAIL B={B:2d} S={S:5d} p={prof} dim={dim_val}  [{desc}]")
    del ctx

print(f"\n{'='*80}")
print("TEST 2: Fresh context per case, WITH infer_shapes() before execute")
print(f"{'='*80}")

for B, S, prof, dim_val, desc in test_cases:
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream(device)

    ctx.set_optimization_profile_async(prof, stream.cuda_stream)
    stream.synchronize()

    d_input_ids     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_attn_mask     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_dims          = torch.full((B,), dim_val, dtype=torch.int64, device=device)
    d_output        = torch.zeros((B, MAX_DIM), dtype=torch.float32, device=device)

    ctx.set_input_shape("input_ids", (B, S))
    ctx.set_tensor_address("input_ids", d_input_ids.data_ptr())
    ctx.set_input_shape("attention_mask", (B, S))
    ctx.set_tensor_address("attention_mask", d_attn_mask.data_ptr())
    ctx.set_input_shape("dimensions", (B,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())
    ctx.set_tensor_address("embeddings", d_output.data_ptr())

    # Call infer_shapes before execute
    try:
        ctx.infer_shapes()
    except Exception:
        pass

    with torch.cuda.stream(stream):
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    if ok:
        norm = torch.norm(d_output).item()
        nz = (d_output.abs() > 1e-12).sum().item()
        print(f"  OK   B={B:2d} S={S:5d} p={prof} dim={dim_val}  "
              f"norm={norm:.4f} nz={nz}/{B*MAX_DIM}  [{desc}]")
    else:
        print(f"  FAIL B={B:2d} S={S:5d} p={prof} dim={dim_val}  [{desc}]")
    del ctx

print(f"\n{'='*80}")
print("TEST 3: REUSE context (like actual code), profile switching")
print(f"{'='*80}")

ctx = engine.create_execution_context()
stream = torch.cuda.Stream(device)

for B, S, prof, dim_val, desc in test_cases:
    stream.synchronize()
    torch.cuda.current_stream(device).synchronize()

    if ctx.active_optimization_profile != prof:
        ctx.set_optimization_profile_async(prof, stream.cuda_stream)
        stream.synchronize()

    d_input_ids     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_attn_mask     = torch.ones((B, S), dtype=torch.int64, device=device)
    d_dims          = torch.full((B,), dim_val, dtype=torch.int64, device=device)
    d_output        = torch.zeros((B, MAX_DIM), dtype=torch.float32, device=device)

    ctx.set_input_shape("input_ids", (B, S))
    ctx.set_tensor_address("input_ids", d_input_ids.data_ptr())
    ctx.set_input_shape("attention_mask", (B, S))
    ctx.set_tensor_address("attention_mask", d_attn_mask.data_ptr())
    ctx.set_input_shape("dimensions", (B,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())
    ctx.set_tensor_address("embeddings", d_output.data_ptr())

    with torch.cuda.stream(stream):
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    if ok:
        norm = torch.norm(d_output).item()
        nz = (d_output.abs() > 1e-12).sum().item()
        print(f"  OK   B={B:2d} S={S:5d} p={prof} dim={dim_val}  "
              f"norm={norm:.4f} nz={nz}/{B*MAX_DIM}  [{desc}]")
    else:
        print(f"  FAIL B={B:2d} S={S:5d} p={prof} dim={dim_val}  [{desc}]")

del ctx

print("\nDone.")
