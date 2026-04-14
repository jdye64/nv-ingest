#!/usr/bin/env python3
"""Quick test: try different 'dimensions' values to find which ones
the TRT embedding engine accepts without overflow."""

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
print(f"TRT: {trt.__version__}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
runtime.engine_host_code_allowed = True

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

device = torch.device("cuda")
torch.cuda.init()

# Use profile 0 (batch 1..32, seq 2..128) for a quick test
ctx = engine.create_execution_context()
stream = torch.cuda.Stream(device)

B, S = 2, 64  # small batch, short seq — fits profile 0 easily

# Create dummy inputs matching engine dtypes (all INT64)
d_input_ids = torch.ones((B, S), dtype=torch.int64, device=device)
d_attention_mask = torch.ones((B, S), dtype=torch.int64, device=device)

ctx.set_input_shape("input_ids", (B, S))
ctx.set_tensor_address("input_ids", d_input_ids.data_ptr())
ctx.set_input_shape("attention_mask", (B, S))
ctx.set_tensor_address("attention_mask", d_attention_mask.data_ptr())

# Test different dimension values
test_values = [0, 1, 2, 128, 384, 512, 768, 1024, 2048, -1]

for dim_val in test_values:
    d_dims = torch.full((B,), dim_val, dtype=torch.int64, device=device)
    ctx.set_input_shape("dimensions", (B,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())

    try:
        ctx.infer_shapes()
    except Exception:
        pass

    try:
        out_shape = tuple(ctx.get_tensor_shape("embeddings"))
    except Exception:
        out_shape = "FAILED"

    if isinstance(out_shape, tuple) and all(d > 0 for d in out_shape):
        d_output = torch.zeros(out_shape, dtype=torch.float32, device=device)
        ctx.set_tensor_address("embeddings", d_output.data_ptr())

        with torch.cuda.stream(stream):
            ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        if ok:
            norm = torch.norm(d_output).item()
            nonzero = (d_output.abs() > 1e-8).sum().item()
            print(f"  dimensions={dim_val:6d}  output_shape={out_shape}  "
                  f"exec=OK  norm={norm:.4f}  nonzero={nonzero}/{d_output.numel()}")
        else:
            print(f"  dimensions={dim_val:6d}  output_shape={out_shape}  exec=FAILED")
    else:
        print(f"  dimensions={dim_val:6d}  output_shape={out_shape}  (shape invalid, skipped exec)")

# Also test with dimensions shape (1,) instead of (B,)
print("\n--- Testing with dimensions shape (1,) instead of (B,) ---")
for dim_val in [0, 384, 1024, 2048]:
    d_dims = torch.full((1,), dim_val, dtype=torch.int64, device=device)
    ctx.set_input_shape("dimensions", (1,))
    ctx.set_tensor_address("dimensions", d_dims.data_ptr())

    try:
        ctx.infer_shapes()
    except Exception:
        pass

    try:
        out_shape = tuple(ctx.get_tensor_shape("embeddings"))
    except Exception:
        out_shape = "FAILED"

    if isinstance(out_shape, tuple) and all(d > 0 for d in out_shape):
        d_output = torch.zeros(out_shape, dtype=torch.float32, device=device)
        ctx.set_tensor_address("embeddings", d_output.data_ptr())

        with torch.cuda.stream(stream):
            ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        if ok:
            norm = torch.norm(d_output).item()
            nonzero = (d_output.abs() > 1e-8).sum().item()
            print(f"  dimensions={dim_val:6d} shape=(1,)  output_shape={out_shape}  "
                  f"exec=OK  norm={norm:.4f}  nonzero={nonzero}/{d_output.numel()}")
        else:
            print(f"  dimensions={dim_val:6d} shape=(1,)  output_shape={out_shape}  exec=FAILED")
    else:
        print(f"  dimensions={dim_val:6d} shape=(1,)  output_shape={out_shape}  (shape invalid)")

print("\nDone.")
