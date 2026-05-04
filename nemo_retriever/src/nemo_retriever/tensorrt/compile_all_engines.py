#!/usr/bin/env python3
"""Compile all YOLOX detection models to TensorRT engines.

Exports each model from PyTorch -> ONNX -> TensorRT .engine, targeting the GPU
architecture of the current machine.  Designed to be run once after install so
that inference can use pre-compiled engines instead of PyTorch.

The three YOLOX models that share the same architecture and input shape:
  - nemotron-page-elements-v3   (document layout detection)
  - nemotron-graphic-elements-v1 (chart element detection)
  - nemotron-table-structure-v1  (table cell/row/column detection)

Usage (standalone):
    python -m nemo_retriever.tensorrt.compile_all_engines [--output-dir ./engines]

Usage (via installed entry point, after ``uv sync --extra local``):
    retriever-build-engines [--output-dir ./engines] [--fp16] [--fp4]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class YOLOXModelSpec:
    """Specification for a YOLOX model to compile."""

    name: str
    package_import: str
    define_model_func: str
    model_key: str
    input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1024)
    onnx_filename: str = ""
    engine_filename: str = ""
    class_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        slug = self.name.replace(" ", "_").lower()
        if not self.onnx_filename:
            self.onnx_filename = f"{slug}.onnx"
        if not self.engine_filename:
            self.engine_filename = f"{slug}.engine"


ALL_MODELS: list[YOLOXModelSpec] = [
    YOLOXModelSpec(
        name="page_elements_v3",
        package_import="nemotron_page_elements_v3.model",
        define_model_func="define_model",
        model_key="page_element_v3",
        class_labels=["table", "chart", "title", "infographic", "paragraph", "header_footer"],
    ),
    YOLOXModelSpec(
        name="graphic_elements_v1",
        package_import="nemotron_graphic_elements_v1.model",
        define_model_func="define_model",
        model_key="Nemotron Graphic Elements v1",
        class_labels=[
            "chart_title",
            "x_title",
            "y_title",
            "xlabel",
            "ylabel",
            "legend_title",
            "legend_label",
            "mark_label",
            "value_label",
            "other",
        ],
    ),
    YOLOXModelSpec(
        name="table_structure_v1",
        package_import="nemotron_table_structure_v1.model",
        define_model_func="define_model",
        model_key="table_structure_v1",
        class_labels=["cell", "row", "column"],
    ),
]


def _load_model(spec: YOLOXModelSpec) -> Any:
    """Dynamically import and instantiate a YOLOX model from its package."""
    import importlib

    mod = importlib.import_module(spec.package_import)
    define_fn: Callable = getattr(mod, spec.define_model_func)
    return define_fn(spec.model_key)


def _export_onnx(
    spec: YOLOXModelSpec,
    output_path: Path,
    *,
    opset: int = 17,
    dynamic_batch: bool = False,
    validate: bool = True,
) -> bool:
    """Export a single YOLOX model to ONNX. Returns True on success."""
    import torch

    print(f"\n{'─' * 60}")
    print(f"  Exporting {spec.name} → ONNX")
    print(f"{'─' * 60}")

    try:
        model = _load_model(spec)
    except ImportError as e:
        print(f"  SKIP: package not installed ({e})")
        return False
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return False

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # All three YOLOX wrappers expose the inner backbone+neck+head as
    # ``model.model``.  Export that rather than the full wrapper (which
    # includes NMS / postprocessing with dynamic output shapes).
    export_model = model.model if hasattr(model, "model") else model
    dummy_input = torch.randint(0, 256, spec.input_shape, dtype=torch.float32, device=device)

    print(f"  Model type: {type(export_model).__name__}")
    print(f"  Input shape: {spec.input_shape}")

    with torch.inference_mode():
        test_out = export_model(dummy_input)
        if isinstance(test_out, (list, tuple)):
            output_names = [f"output_{i}" for i in range(len(test_out))]
        elif isinstance(test_out, dict):
            output_names = list(test_out.keys())
        else:
            output_names = ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch_size"}}
        for oname in output_names:
            dynamic_axes[oname] = {0: "batch_size"}

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    if validate:
        try:
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(str(output_path), providers=providers)
            input_np = dummy_input.cpu().numpy()
            ort_inputs = {session.get_inputs()[0].name: input_np}
            ort_outputs = session.run(None, ort_inputs)
            print(f"  Validation: PASSED ({len(ort_outputs)} output(s))")

            with torch.inference_mode():
                pt_out = export_model(dummy_input)
            if isinstance(pt_out, torch.Tensor):
                diff = np.abs(pt_out.cpu().numpy() - ort_outputs[0]).max()
                print(f"  Max abs diff (PT vs ORT): {diff:.6f}")
        except ImportError:
            print("  Validation skipped (onnx/onnxruntime not installed)")

    del model, export_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = True,
    fp4: bool = False,
    workspace_gib: float = 4.0,
    max_batch: int = 16,
) -> bool:
    """Build a TensorRT engine from an ONNX file. Returns True on success."""
    try:
        import tensorrt as trt
    except ImportError:
        print("  ERROR: tensorrt not installed")
        print("         pip install tensorrt --extra-index-url https://pypi.nvidia.com")
        return False

    print("  Building TensorRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error [{i}]: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))

    precision_parts: list[str] = []
    if fp4:
        if hasattr(trt.BuilderFlag, "FP4"):
            config.set_flag(trt.BuilderFlag.FP4)
            precision_parts.append("FP4")
        else:
            print("  WARNING: FP4 not available in this TensorRT version, falling back to FP16")
            fp16 = True

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        precision_parts.append("FP16")
    elif fp16:
        precision_parts.append("FP32 (FP16 not supported on this GPU)")

    if not precision_parts:
        precision_parts.append("FP32")

    print(f"  Precision: {' + '.join(precision_parts)}")

    input_tensor = network.get_input(0)
    if input_tensor.shape[0] == -1:
        profile = builder.create_optimization_profile()
        _, c, h, w = input_tensor.shape
        opt_batch = min(8, max_batch)
        profile.set_shape(
            input_tensor.name,
            min=(1, c, h, w),
            opt=(opt_batch, c, h, w),
            max=(max_batch, c, h, w),
        )
        config.add_optimization_profile(profile)
        print(f"  Dynamic batch: min=1, opt={opt_batch}, max={max_batch}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  ERROR: TensorRT engine build failed")
        return False

    engine_path.write_bytes(serialized)
    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB)")
    return True


def compile_all(
    output_dir: Path,
    *,
    models: Optional[list[str]] = None,
    fp16: bool = True,
    fp4: bool = False,
    opset: int = 17,
    workspace_gib: float = 4.0,
    keep_onnx: bool = False,
    validate: bool = True,
    dynamic_batch: bool = True,
    max_batch: int = 16,
) -> dict[str, bool]:
    """Compile all (or selected) YOLOX models. Returns ``{name: success}`` map."""
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = ALL_MODELS
    if models:
        requested = {m.lower().replace("-", "_") for m in models}
        specs = [s for s in specs if s.name.lower().replace("-", "_") in requested]
        if not specs:
            print(f"ERROR: No matching models for {models}")
            print(f"Available: {[s.name for s in ALL_MODELS]}")
            return {}

    results: dict[str, bool] = {}

    for spec in specs:
        t0 = time.perf_counter()
        onnx_path = output_dir / spec.onnx_filename
        engine_path = output_dir / spec.engine_filename

        ok = _export_onnx(
            spec,
            onnx_path,
            opset=opset,
            dynamic_batch=dynamic_batch,
            validate=validate,
        )
        if not ok:
            results[spec.name] = False
            continue

        ok = _build_engine(
            onnx_path,
            engine_path,
            fp16=fp16,
            fp4=fp4,
            workspace_gib=workspace_gib,
            max_batch=max_batch,
        )
        results[spec.name] = ok

        if ok and not keep_onnx:
            onnx_path.unlink(missing_ok=True)
            print("  Cleaned up intermediate ONNX file")

        elapsed = time.perf_counter() - t0
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {spec.name} ({elapsed:.1f}s)")

    return results


def _detect_gpu_info() -> str:
    """Best-effort GPU description for the summary banner."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} (SM {props.major}.{props.minor}, {props.total_mem // (1 << 20)} MB)"
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile all YOLOX detection models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Models compiled:
  page_elements_v3      Document layout detection (tables, charts, titles, ...)
  graphic_elements_v1   Chart element detection (axes, labels, legends, ...)
  table_structure_v1    Table structure detection (cells, rows, columns)

TensorRT engines are architecture-specific: they must be compiled on the
target GPU (or a GPU with the same SM version).
""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path.home() / ".cache" / "nemo-retriever" / "engines",
        help="Directory for output .engine files (default: ~/.cache/nemo-retriever/engines)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=None,
        help="Specific models to compile (default: all). "
        "Choices: page_elements_v3, graphic_elements_v1, table_structure_v1",
    )
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16, use FP32")
    parser.add_argument("--fp4", action="store_true", help="Enable FP4 for Blackwell GPUs (TensorRT >= 10.8)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--workspace", type=float, default=4.0, help="TRT workspace size in GiB (default: 4)")
    parser.add_argument("--keep-onnx", action="store_true", help="Keep intermediate ONNX files")
    parser.add_argument("--no-validate", action="store_true", help="Skip ONNX validation step")
    parser.add_argument(
        "--dynamic-batch", action="store_true", default=True,
        help="Enable dynamic batch dimension (default: on)",
    )
    parser.add_argument("--no-dynamic-batch", action="store_false", dest="dynamic_batch", help="Disable dynamic batch")
    parser.add_argument("--max-batch", type=int, default=16, help="Max batch size for dynamic-batch engines (default: 16)")
    args = parser.parse_args()

    gpu = _detect_gpu_info()
    print("=" * 60)
    print("  Nemo Retriever — TensorRT Engine Compilation")
    print("=" * 60)
    print(f"  GPU:        {gpu}")
    print(f"  Output:     {args.output_dir}")
    precision = "FP4" if args.fp4 else ("FP16" if args.fp16 else "FP32")
    print(f"  Precision:  {precision}")
    if args.models:
        print(f"  Models:     {', '.join(args.models)}")
    else:
        print(f"  Models:     all ({len(ALL_MODELS)})")
    print()

    t_start = time.perf_counter()
    results = compile_all(
        args.output_dir,
        models=args.models,
        fp16=args.fp16,
        fp4=args.fp4,
        opset=args.opset,
        workspace_gib=args.workspace,
        keep_onnx=args.keep_onnx,
        validate=not args.no_validate,
        dynamic_batch=args.dynamic_batch,
        max_batch=args.max_batch,
    )
    t_total = time.perf_counter() - t_start

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({t_total:.1f}s total)")
    print(f"{'=' * 60}")
    succeeded = 0
    failed = 0
    skipped = 0
    for name in [s.name for s in ALL_MODELS]:
        if name not in results:
            if args.models and name.lower().replace("-", "_") not in {m.lower().replace("-", "_") for m in args.models}:
                continue
            print(f"  {name:<30} SKIPPED (not requested or dep missing)")
            skipped += 1
        elif results[name]:
            engine_path = args.output_dir / f"{name}.engine"
            size = engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else 0
            print(f"  {name:<30} OK ({size:.1f} MB)")
            succeeded += 1
        else:
            print(f"  {name:<30} FAILED")
            failed += 1

    print(f"\n  {succeeded} succeeded, {failed} failed, {skipped} skipped")
    if succeeded > 0:
        print(f"\n  Engine files written to: {args.output_dir}")
        print(f"  These engines are compiled for: {gpu}")
        print(f"  They will only work on GPUs with the same SM architecture.")
    print("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
