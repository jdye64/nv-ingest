#!/usr/bin/env python3
"""Inspect ALL TRT engines used by the pipeline to see exact tensor dtypes, shapes, and profile ranges."""

import sys
from pathlib import Path

import tensorrt as trt
import numpy as np

print(f"TensorRT version: {trt.__version__}\n")

NGC_MODELS = Path("/datasets/nv-ingest/models")

ENGINES = {
    "page-elements": NGC_MODELS / "page-elements/ngc/hub/models--nim--nvidia--nemotron-page-elements-v3",
    "table-structure": NGC_MODELS / "table-structure/ngc/hub/models--nim--nvidia--nemotron-table-structure-v1",
    "graphic-elements": NGC_MODELS / "graphic-elements/ngc/hub/models--nim--nvidia--nemotron-graphic-elements-v1",
    "ocr": NGC_MODELS / "ocr",
    "embedding": NGC_MODELS / "embedding/ngc/hub/models--nim--nvidia--llama-nemotron-embed-1b-v2",
}


def find_engine(base: Path) -> Path:
    """Return the largest file under *base* (likely the engine)."""
    best, best_size = None, 0
    for f in base.rglob("*"):
        if f.is_file() and f.stat().st_size > best_size:
            best, best_size = f, f.stat().st_size
    return best  # type: ignore


def inspect_engine(label: str, engine_path: Path) -> None:
    print(f"{'#' * 80}")
    print(f"#  {label}")
    print(f"{'#' * 80}")

    if not engine_path.exists():
        print(f"  SKIPPED — path does not exist: {engine_path}\n")
        return

    resolved = find_engine(engine_path)
    if resolved is None:
        print(f"  SKIPPED — no files found under: {engine_path}\n")
        return

    size_mb = resolved.stat().st_size / 1e6
    print(f"  Path:      {engine_path}")
    print(f"  Engine:    {resolved}")
    print(f"  Size:      {size_mb:.1f} MB")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    runtime.engine_host_code_allowed = True

    try:
        with open(resolved, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"  ERROR deserializing: {e}\n")
        return

    if engine is None:
        print(f"  ERROR: deserialize returned None\n")
        return

    print(f"  IO tensors:    {engine.num_io_tensors}")
    print(f"  Profiles:      {engine.num_optimization_profiles}")
    print(f"  Layers:        {engine.num_layers}")
    print()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        trt_dtype = engine.get_tensor_dtype(name)
        np_dtype = trt.nptype(trt_dtype)
        shape = tuple(engine.get_tensor_shape(name))
        is_input = mode == trt.TensorIOMode.INPUT
        tag = "INPUT " if is_input else "OUTPUT"

        print(f"  [{i}] {tag}  {name}")
        print(f"       trt_dtype={trt_dtype}  np_dtype={np.dtype(np_dtype)}  shape={shape}")

        if is_input:
            for p_idx in range(engine.num_optimization_profiles):
                try:
                    mn, opt, mx = engine.get_tensor_profile_shape(name, p_idx)
                    print(f"       profile[{p_idx}]: min={tuple(mn)}  opt={tuple(opt)}  max={tuple(mx)}")
                except Exception as e:
                    print(f"       profile[{p_idx}]: ERROR: {e}")

    print()


for label, path in ENGINES.items():
    inspect_engine(label, path)

print("Done.")
