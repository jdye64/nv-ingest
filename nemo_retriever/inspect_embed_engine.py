#!/usr/bin/env python3
"""Inspect the TRT embedding engine to see exact tensor dtypes, shapes, and profile ranges."""

import sys
from pathlib import Path

NGC_MODELS = Path("/datasets/nv-ingest/models")
raw_path = NGC_MODELS / "embedding/ngc/hub/models--nim--nvidia--llama-nemotron-embed-1b-v2"

# Walk the directory to find the largest file (likely the engine)
def find_engine(base: Path) -> Path:
    best, best_size = None, 0
    for f in base.rglob("*"):
        if f.is_file() and f.stat().st_size > best_size:
            best, best_size = f, f.stat().st_size
    return best  # type: ignore

resolved = str(find_engine(raw_path))
print(f"Raw path: {raw_path}")
print(f"Resolved engine path: {resolved}")
print(f"File size: {Path(resolved).stat().st_size / 1e6:.1f} MB")

import tensorrt as trt
import numpy as np

print(f"\nTensorRT version: {trt.__version__}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
runtime.engine_host_code_allowed = True

with open(resolved, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    print("ERROR: Failed to deserialize engine!")
    sys.exit(1)

print(f"\nEngine loaded successfully")
print(f"  num_io_tensors: {engine.num_io_tensors}")
print(f"  num_optimization_profiles: {engine.num_optimization_profiles}")
print(f"  num_layers: {engine.num_layers}")

print(f"\n{'='*80}")
print("TENSOR DETAILS")
print(f"{'='*80}")

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    trt_dtype = engine.get_tensor_dtype(name)
    np_dtype = trt.nptype(trt_dtype)
    shape = engine.get_tensor_shape(name)
    is_input = mode == trt.TensorIOMode.INPUT

    print(f"\n  [{i}] name={name}")
    print(f"      mode={'INPUT' if is_input else 'OUTPUT'}")
    print(f"      trt_dtype={trt_dtype}  (raw enum: {int(trt_dtype)})")
    print(f"      np_dtype={np_dtype}  (np.dtype: {np.dtype(np_dtype)})")
    print(f"      static_shape={tuple(shape)}  ndim={len(shape)}")

    if is_input:
        for p_idx in range(engine.num_optimization_profiles):
            try:
                mn, opt, mx = engine.get_tensor_profile_shape(name, p_idx)
                print(f"      profile[{p_idx}]: min={tuple(mn)} opt={tuple(opt)} max={tuple(mx)}")
            except Exception as e:
                print(f"      profile[{p_idx}]: ERROR: {e}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    trt_dtype = engine.get_tensor_dtype(name)
    np_dtype = trt.nptype(trt_dtype)
    mode = engine.get_tensor_mode(name)
    is_input = mode == trt.TensorIOMode.INPUT
    tag = "IN " if is_input else "OUT"
    print(f"  {tag} {name:25s} trt={str(trt_dtype):20s} np={str(np.dtype(np_dtype)):10s} shape={tuple(engine.get_tensor_shape(name))}")
