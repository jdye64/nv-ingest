#!/usr/bin/env python3
"""
Stage-by-stage pipeline runner with disk persistence.

Runs individual DAG stages (or the full pipeline) with checkpointing
to disk between each stage, enabling manual inspection of intermediate
results and isolated performance benchmarking.

Usage
-----
    # Run the full PDF pipeline, persisting every intermediate stage
    python nemo_retriever/stage_runner.py run-all /path/to/docs/*.pdf \\
        --output-path /tmp/stages --extract-text --extract-images

    # Re-run a single stage from a previously saved upstream output
    python nemo_retriever/stage_runner.py run pdf_extraction \\
        --input-dir /tmp/stages/02_pdf_split \\
        --output-path /tmp/stages

    # Benchmark a single stage (N iterations, timing report)
    python nemo_retriever/stage_runner.py benchmark pdf_extraction \\
        --input-dir /tmp/stages/02_pdf_split \\
        --iterations 10

Each stage directory contains:
    dataframe.pkl   – pickled DataFrame for faithful stage-to-stage replay
    stage_meta.json – timing, row count, column list, actor kwargs
    assets/         – human-inspectable images + per-row JSON metadata
"""

from __future__ import annotations

import argparse
import gc
import glob as _glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import pandas as pd

from nemo_retriever.utils.stage_io import load_stage, save_stage


# ═══════════════════════════════════════════════════════════════════════════
# Stage registry
# ═══════════════════════════════════════════════════════════════════════════


class StageSpec(NamedTuple):
    """Descriptor for a single pipeline stage."""

    index: int
    dir_name: str
    actor_factory: Callable[..., Any]
    default_kwargs: Dict[str, Any]


def _make_file_load_actor(**_kwargs: Any) -> None:
    """Sentinel — file_load has no actor; handled specially."""
    return None


def _make_doc_to_pdf_actor(**kwargs: Any) -> Any:
    from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor

    return DocToPdfConversionActor(**kwargs)


def _make_pdf_split_actor(**kwargs: Any) -> Any:
    from nemo_retriever.params import PdfSplitParams
    from nemo_retriever.pdf.split import PDFSplitActor

    return PDFSplitActor(split_params=PdfSplitParams(**kwargs))


def _make_pdf_extraction_actor(**kwargs: Any) -> Any:
    from nemo_retriever.pdf.extract import PDFExtractionActor

    return PDFExtractionActor(**kwargs)


def _make_page_elements_actor(**kwargs: Any) -> Any:
    from nemo_retriever.page_elements.page_elements import PageElementDetectionActor

    return PageElementDetectionActor(**kwargs)


def _make_table_structure_actor(**kwargs: Any) -> Any:
    from nemo_retriever.table.table_detection import TableStructureActor

    return TableStructureActor(**kwargs)


def _make_graphic_elements_actor(**kwargs: Any) -> Any:
    from nemo_retriever.chart.chart_detection import GraphicElementsActor

    return GraphicElementsActor(**kwargs)


def _make_ocr_actor(**kwargs: Any) -> Any:
    from nemo_retriever.ocr.ocr import OCRActor

    return OCRActor(**kwargs)


def _make_text_chunk_actor(**kwargs: Any) -> Any:
    from nemo_retriever.params import TextChunkParams
    from nemo_retriever.txt.ray_data import TextChunkActor

    return TextChunkActor(params=TextChunkParams(**kwargs))


def _make_embed_actor(**kwargs: Any) -> Any:
    from nemo_retriever.params import EmbedParams
    from nemo_retriever.text_embed.operators import _BatchEmbedActor

    return _BatchEmbedActor(params=EmbedParams(**kwargs))


# Ordered registry for the default PDF pipeline.
STAGE_REGISTRY: Dict[str, StageSpec] = {
    "file_load": StageSpec(
        index=0,
        dir_name="00_file_load",
        actor_factory=_make_file_load_actor,
        default_kwargs={},
    ),
    "doc_to_pdf": StageSpec(
        index=1,
        dir_name="01_doc_to_pdf",
        actor_factory=_make_doc_to_pdf_actor,
        default_kwargs={},
    ),
    "pdf_split": StageSpec(
        index=2,
        dir_name="02_pdf_split",
        actor_factory=_make_pdf_split_actor,
        default_kwargs={},
    ),
    "pdf_extraction": StageSpec(
        index=3,
        dir_name="03_pdf_extraction",
        actor_factory=_make_pdf_extraction_actor,
        default_kwargs={
            "extract_text": True,
            "extract_images": True,
            "extract_tables": True,
            "extract_charts": True,
            "dpi": 200,
            "image_format": "jpeg",
        },
    ),
    "page_elements": StageSpec(
        index=4,
        dir_name="04_page_elements",
        actor_factory=_make_page_elements_actor,
        default_kwargs={},
    ),
    "table_structure": StageSpec(
        index=5,
        dir_name="05_table_structure",
        actor_factory=_make_table_structure_actor,
        default_kwargs={},
    ),
    "graphic_elements": StageSpec(
        index=6,
        dir_name="06_graphic_elements",
        actor_factory=_make_graphic_elements_actor,
        default_kwargs={},
    ),
    "ocr": StageSpec(
        index=7,
        dir_name="07_ocr",
        actor_factory=_make_ocr_actor,
        default_kwargs={},
    ),
    "text_chunk": StageSpec(
        index=8,
        dir_name="08_text_chunk",
        actor_factory=_make_text_chunk_actor,
        default_kwargs={},
    ),
    "embed": StageSpec(
        index=9,
        dir_name="09_embed",
        actor_factory=_make_embed_actor,
        default_kwargs={},
    ),
}

ORDERED_STAGES: List[str] = sorted(STAGE_REGISTRY, key=lambda s: STAGE_REGISTRY[s].index)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_files(paths: List[str]) -> pd.DataFrame:
    """Read files as raw bytes into a DataFrame with ``bytes`` and ``path`` columns."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            rows.append({"bytes": fp.read_bytes(), "path": str(fp.resolve())})
        else:
            print(f"  Skipping {p} (not a file)", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["bytes", "path"])
    return pd.DataFrame(rows)


def _expand_globs(patterns: List[str]) -> List[str]:
    expanded: List[str] = []
    for pat in patterns:
        matches = sorted(_glob.glob(pat))
        expanded.extend(matches if matches else [pat])
    return expanded


def _run_actor(actor: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Run an actor on a DataFrame and normalise the result."""
    result = actor.run(df)
    if isinstance(result, list):
        result = pd.DataFrame(result)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"Actor returned {type(result).__name__}, expected DataFrame")
    return result


def _unload_actor(actor: Any) -> None:
    """Delete an actor and release its GPU memory.

    Actors load models eagerly in __init__ but have no cleanup methods.
    We explicitly delete the object, run a full GC pass, and clear the
    CUDA memory cache so VRAM is available for subsequent stages.
    """
    del actor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _coerce_value(val_str: str) -> Any:
    """Best-effort type coercion for CLI values."""
    if val_str.lower() in ("true", "false"):
        return val_str.lower() == "true"
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        return val_str


def _parse_extra_kwargs(raw: Optional[List[str]]) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Parse ``--kwarg key=value`` pairs.

    Supports stage-prefixed kwargs like ``pdf_extraction.dpi=300`` which
    target a specific stage, and unprefixed kwargs that apply globally.

    Returns
    -------
    global_kwargs : dict
        Kwargs with no stage prefix.
    stage_kwargs : dict[str, dict]
        Mapping of stage_name -> kwargs for that stage only.
    """
    if not raw:
        return {}, {}
    global_kw: Dict[str, Any] = {}
    stage_kw: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if "=" not in item:
            print(f"  WARNING: ignoring malformed --kwarg {item!r} (expected key=value)", file=sys.stderr)
            continue
        key, val_str = item.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()
        val = _coerce_value(val_str)
        if "." in key:
            stage, param = key.split(".", 1)
            stage_kw.setdefault(stage, {})[param] = val
        else:
            global_kw[key] = val
    return global_kw, stage_kw


def _kwargs_for_stage(
    stage_name: str,
    defaults: Dict[str, Any],
    global_kw: Dict[str, Any],
    stage_kw: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the final kwargs for a specific stage.

    Global kwargs are only applied when the stage already declares that key
    in its defaults (to avoid passing e.g. ``dpi`` to ``DocToPdfConversionActor``).
    Stage-prefixed kwargs are always applied to their target stage.
    """
    merged = dict(defaults)
    for k, v in global_kw.items():
        if k in defaults:
            merged[k] = v
    if stage_name in stage_kw:
        merged.update(stage_kw[stage_name])
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: run-all
# ═══════════════════════════════════════════════════════════════════════════


def cmd_run_all(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    global_kw, stage_kw = _parse_extra_kwargs(args.kwarg)

    # Resolve which stages to run.
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]
        for name in stage_names:
            if name not in STAGE_REGISTRY:
                print(f"ERROR: unknown stage {name!r}. Choose from: {', '.join(ORDERED_STAGES)}", file=sys.stderr)
                sys.exit(1)
    else:
        stage_names = list(ORDERED_STAGES)

    # Ensure file_load is first when included.
    if "file_load" in stage_names:
        stage_names.remove("file_load")
        stage_names.insert(0, "file_load")

    file_paths = _expand_globs(args.input_files)
    if not file_paths:
        print("ERROR: no input files matched", file=sys.stderr)
        sys.exit(1)

    print(f"Pipeline: {' -> '.join(stage_names)}")
    print(f"Input files: {len(file_paths)}")
    print(f"Output: {output_path}\n")

    pipeline_t0 = time.perf_counter()
    df: Optional[pd.DataFrame] = None

    for stage_name in stage_names:
        spec = STAGE_REGISTRY[stage_name]
        stage_dir = output_path / spec.dir_name
        kwargs = _kwargs_for_stage(stage_name, spec.default_kwargs, global_kw, stage_kw)

        print(f"{'─'*60}")
        print(f"Stage: {stage_name} ({spec.dir_name})")

        t0 = time.perf_counter()

        actor = None
        if stage_name == "file_load":
            df = _load_files(file_paths)
        else:
            if df is None:
                print("  ERROR: no input DataFrame (did file_load run?)", file=sys.stderr)
                sys.exit(1)
            actor = spec.actor_factory(**kwargs)
            df = _run_actor(actor, df)

        elapsed = time.perf_counter() - t0

        if actor is not None:
            _unload_actor(actor)
            actor = None
            print("  Unloaded actor and freed GPU memory")

        meta = {
            "stage": stage_name,
            "elapsed_seconds": round(elapsed, 4),
            "actor_kwargs": kwargs,
        }
        save_stage(df, stage_dir, stage_meta=meta)

        rows = len(df)
        rps = rows / elapsed if elapsed > 0 else float("inf")
        print(f"  {rows} rows in {elapsed:.3f}s ({rps:.1f} rows/s)")
        print(f"  Saved to {stage_dir}")

    total = time.perf_counter() - pipeline_t0
    print(f"\n{'═'*60}")
    print(f"Pipeline complete in {total:.2f}s")
    print(f"Output: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: run
# ═══════════════════════════════════════════════════════════════════════════


def cmd_run(args: argparse.Namespace) -> None:
    stage_name = args.stage
    if stage_name not in STAGE_REGISTRY:
        print(f"ERROR: unknown stage {stage_name!r}. Choose from: {', '.join(ORDERED_STAGES)}", file=sys.stderr)
        sys.exit(1)
    if stage_name == "file_load":
        print("ERROR: file_load has no actor; use run-all to start from files", file=sys.stderr)
        sys.exit(1)

    spec = STAGE_REGISTRY[stage_name]
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    global_kw, stage_kw = _parse_extra_kwargs(args.kwarg)
    kwargs = _kwargs_for_stage(stage_name, spec.default_kwargs, global_kw, stage_kw)

    print(f"Stage: {stage_name}")
    print(f"Input: {input_dir}")
    print(f"Kwargs: {json.dumps(kwargs, default=str)}")

    df = load_stage(input_dir)
    print(f"Loaded {len(df)} rows from {input_dir / 'dataframe.pkl'}")

    actor = spec.actor_factory(**kwargs)
    t0 = time.perf_counter()
    df_out = _run_actor(actor, df)
    elapsed = time.perf_counter() - t0

    stage_dir = output_path / spec.dir_name
    meta = {
        "stage": stage_name,
        "elapsed_seconds": round(elapsed, 4),
        "input_dir": str(input_dir),
        "actor_kwargs": kwargs,
    }
    save_stage(df_out, stage_dir, stage_meta=meta)

    rows = len(df_out)
    rps = rows / elapsed if elapsed > 0 else float("inf")
    print(f"\n{rows} rows in {elapsed:.3f}s ({rps:.1f} rows/s)")
    print(f"Saved to {stage_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: benchmark
# ═══════════════════════════════════════════════════════════════════════════


def cmd_benchmark(args: argparse.Namespace) -> None:
    stage_name = args.stage
    if stage_name not in STAGE_REGISTRY:
        print(f"ERROR: unknown stage {stage_name!r}. Choose from: {', '.join(ORDERED_STAGES)}", file=sys.stderr)
        sys.exit(1)
    if stage_name == "file_load":
        print("ERROR: file_load has no actor to benchmark", file=sys.stderr)
        sys.exit(1)

    spec = STAGE_REGISTRY[stage_name]
    input_dir = Path(args.input_dir)
    iterations = max(args.iterations, 1)
    global_kw, stage_kw = _parse_extra_kwargs(args.kwarg)
    kwargs = _kwargs_for_stage(stage_name, spec.default_kwargs, global_kw, stage_kw)

    print(f"Benchmark: {stage_name} x {iterations} iterations")
    print(f"Input: {input_dir}")
    print(f"Kwargs: {json.dumps(kwargs, default=str)}\n")

    df = load_stage(input_dir)
    n_rows = len(df)
    print(f"Loaded {n_rows} rows\n")

    timings: List[float] = []
    for i in range(iterations):
        actor = spec.actor_factory(**kwargs)
        t0 = time.perf_counter()
        _ = _run_actor(actor, df)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        rps = n_rows / elapsed if elapsed > 0 else float("inf")
        print(f"  iter {i+1:3d}/{iterations}: {elapsed:.4f}s  ({rps:.1f} rows/s)")

    print(f"\n{'═'*60}")
    print(f"Benchmark results: {stage_name}")
    print(f"{'─'*60}")
    print(f"  Rows per iteration : {n_rows}")
    print(f"  Iterations         : {iterations}")
    print(f"  Min                : {min(timings):.4f}s  ({n_rows / min(timings):.1f} rows/s)")
    print(f"  Max                : {max(timings):.4f}s  ({n_rows / max(timings):.1f} rows/s)")
    print(f"  Mean               : {statistics.mean(timings):.4f}s  ({n_rows / statistics.mean(timings):.1f} rows/s)")
    if len(timings) > 1:
        print(f"  Stdev              : {statistics.stdev(timings):.4f}s")
        med = statistics.median(timings)
        print(f"  Median             : {med:.4f}s  ({n_rows / med:.1f} rows/s)")
    print(f"{'═'*60}")

    if args.output_json:
        results = {
            "stage": stage_name,
            "rows": n_rows,
            "iterations": iterations,
            "kwargs": kwargs,
            "timings_seconds": [round(t, 6) for t in timings],
            "min_seconds": round(min(timings), 6),
            "max_seconds": round(max(timings), 6),
            "mean_seconds": round(statistics.mean(timings), 6),
            "mean_rows_per_second": round(n_rows / statistics.mean(timings), 2),
        }
        out = Path(args.output_json)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: list
# ═══════════════════════════════════════════════════════════════════════════


def cmd_list(_args: argparse.Namespace) -> None:
    print(f"{'Index':<7} {'Name':<20} {'Directory':<25}")
    print(f"{'─'*7} {'─'*20} {'─'*25}")
    for name in ORDERED_STAGES:
        spec = STAGE_REGISTRY[name]
        print(f"{spec.index:<7} {name:<20} {spec.dir_name:<25}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI wiring
# ═══════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-by-stage pipeline runner with disk persistence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run-all ───────────────────────────────────────────────────────
    p_all = sub.add_parser(
        "run-all",
        help="Run the full pipeline, persisting every intermediate stage.",
    )
    p_all.add_argument("input_files", nargs="+", help="Input file paths / globs")
    p_all.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", "."),
        help="Base output directory (default: $OUTPUT_PATH or cwd)",
    )
    p_all.add_argument(
        "--stages",
        default=None,
        help="Comma-separated list of stages to run (default: all)",
    )
    p_all.add_argument(
        "--kwarg",
        action="append",
        metavar="KEY=VALUE",
        help="Override actor kwargs (repeatable, e.g. --kwarg dpi=300)",
    )
    p_all.set_defaults(func=cmd_run_all)

    # ── run ───────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Run a single stage from its upstream output directory.",
    )
    p_run.add_argument("stage", help=f"Stage name ({', '.join(ORDERED_STAGES)})")
    p_run.add_argument(
        "--input-dir",
        required=True,
        help="Path to upstream stage output directory (containing dataframe.pkl)",
    )
    p_run.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", "."),
        help="Base output directory (default: $OUTPUT_PATH or cwd)",
    )
    p_run.add_argument(
        "--kwarg",
        action="append",
        metavar="KEY=VALUE",
        help="Override actor kwargs (repeatable)",
    )
    p_run.set_defaults(func=cmd_run)

    # ── benchmark ─────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "benchmark",
        help="Benchmark a single stage with N iterations.",
    )
    p_bench.add_argument("stage", help=f"Stage name ({', '.join(ORDERED_STAGES)})")
    p_bench.add_argument(
        "--input-dir",
        required=True,
        help="Path to upstream stage output directory",
    )
    p_bench.add_argument("--iterations", type=int, default=5, help="Number of iterations (default: 5)")
    p_bench.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write benchmark results as JSON",
    )
    p_bench.add_argument(
        "--kwarg",
        action="append",
        metavar="KEY=VALUE",
        help="Override actor kwargs (repeatable)",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # ── list ──────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List all registered stages.")
    p_list.set_defaults(func=cmd_list)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
