# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the ultra-low-latency PDF processing experiment."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Ultra-low-latency fused PDF processing experiment.")


def _find_pdfs(input_dir: str, limit: Optional[int] = None) -> list[Path]:
    d = Path(input_dir)
    if not d.is_dir():
        raise typer.BadParameter(f"Not a directory: {input_dir}")
    pdfs = sorted(d.glob("**/*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise typer.BadParameter(f"No PDF files found in {input_dir}")
    return pdfs


def _gpu_mem_mb() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            resv = torch.cuda.memory_reserved() / 1024**2
            return f"{alloc:.0f} MB allocated / {resv:.0f} MB reserved"
    except Exception:
        pass
    return "N/A"


# ─── Fused pipeline run ─────────────────────────────────────────────


def _run_fused(
    pdfs: list[Path],
    device: str,
    warmup_runs: int,
    console: Console,
) -> list:
    from nemo_retriever.experiment.fused_pipeline import FusedPDFPipeline

    console.print(
        Panel(
            "[bold]Fused Pipeline — Optimized[/bold]\n"
            "Single process · all models on one GPU · cuDNN benchmark · "
            "batched YOLOX · GPU-native crops · TF32 · no HTTP/Redis/base64",
            style="green",
        )
    )

    pipeline = FusedPDFPipeline(device=device)

    # Load models
    console.print("\n[bold]Loading models …[/bold]")
    load_times = pipeline.load_models()
    for stage, secs in load_times.items():
        console.print(f"  {stage}: {secs:.1f}s")
    console.print(f"  GPU memory: {_gpu_mem_mb()}")

    # Warmup
    console.print(f"\n[bold]Warming up ({warmup_runs} runs) …[/bold]")
    wu = pipeline.warmup(runs=warmup_runs)
    console.print(f"  Warmup: {wu:.2f}s\n")

    # Process each PDF
    results = []
    for pdf_path in pdfs:
        result = pipeline.process_pdf(str(pdf_path))
        results.append(result)

        # Per-PDF timing table
        name = pdf_path.name
        t = result.timings
        n_struct = sum(pr.num_structured_regions for pr in result.page_results)
        n_text = len(result.all_texts)
        emb_shape = tuple(result.embeddings.shape) if result.embeddings is not None else "(none)"

        tbl = Table(
            title=f"{name}  ({result.num_pages} pages, {n_struct} structured regions)",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Stage", style="white", min_width=24)
        tbl.add_column("Time (ms)", justify="right", style="yellow")

        for stage, ms in t.items():
            style = "bold green" if stage == "total_ms" else ""
            label = stage.replace("_ms", "").replace("_", " ")
            tbl.add_row(label, f"{ms:.1f}", style=style)
        tbl.add_row("texts embedded", str(n_text))
        tbl.add_row("embedding shape", str(emb_shape))
        console.print(tbl)
        console.print()

    return results


# ─── Baseline pipeline run ───────────────────────────────────────────


def _run_baseline(pdfs: list[Path], console: Console) -> list[dict]:
    from nemo_retriever.ingest_modes.inprocess import (
        InProcessIngestor,
        pdf_to_pages_df,
        run_pipeline_tasks_on_df,
    )

    console.print(
        Panel(
            "[bold]Baseline Pipeline — Standard Inprocess[/bold]\n"
            "Sequential render · per-page YOLOX · PIL crop · base64 encoding · "
            "DataFrame between stages",
            style="yellow",
        )
    )

    # Build the task list (loads models once)
    console.print("\n[bold]Setting up baseline (loading models) …[/bold]")
    t0 = time.perf_counter()
    ingestor = InProcessIngestor()
    ingestor.files([str(pdfs[0])])
    ingestor.extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    ingestor.embed()
    setup_s = time.perf_counter() - t0
    console.print(f"  Setup (incl. model load): {setup_s:.1f}s")
    console.print(f"  GPU memory: {_gpu_mem_mb()}\n")

    per_doc_tasks, _post_tasks = ingestor.get_pipeline_tasks()

    baseline_results = []
    for pdf_path in pdfs:
        name = pdf_path.name
        t0 = time.perf_counter()
        initial_df = pdf_to_pages_df(str(pdf_path))
        num_pages = len(initial_df)
        _result_df, metrics = run_pipeline_tasks_on_df(initial_df, per_doc_tasks)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        tbl = Table(
            title=f"{name}  ({num_pages} pages)",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Stage", style="white", min_width=24)
        tbl.add_column("Time (ms)", justify="right", style="yellow")
        for m in metrics:
            tbl.add_row(m["stage"], f"{m['duration_sec'] * 1000:.1f}")
        tbl.add_row("total", f"{elapsed_ms:.1f}", style="bold yellow")
        console.print(tbl)
        console.print()

        baseline_results.append(
            {
                "name": name,
                "num_pages": num_pages,
                "total_ms": elapsed_ms,
                "metrics": metrics,
            }
        )

    return baseline_results


# ─── Summary + comparison ────────────────────────────────────────────


def _print_summary(
    fused_results: list,
    baseline_results: list[dict] | None,
    console: Console,
) -> None:
    total_pages_f = sum(r.num_pages for r in fused_results)
    total_ms_f = sum(r.timings["total_ms"] for r in fused_results)
    avg_pdf_f = total_ms_f / max(len(fused_results), 1)
    avg_page_f = total_ms_f / max(total_pages_f, 1)
    pps_f = total_pages_f / max(total_ms_f / 1000, 0.001)

    tbl = Table(title="Summary", show_header=True, header_style="bold")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Fused", justify="right", style="green")
    if baseline_results:
        tbl.add_column("Baseline", justify="right", style="yellow")
        tbl.add_column("Speedup", justify="right", style="bold magenta")

    total_pages_b = sum(r["num_pages"] for r in baseline_results) if baseline_results else 0
    total_ms_b = sum(r["total_ms"] for r in baseline_results) if baseline_results else 0
    avg_pdf_b = total_ms_b / max(len(baseline_results), 1) if baseline_results else 0
    avg_page_b = total_ms_b / max(total_pages_b, 1) if baseline_results else 0
    pps_b = total_pages_b / max(total_ms_b / 1000, 0.001) if baseline_results else 0

    def _row(label: str, fused_val: str, base_val: str = "", speedup: str = "") -> None:
        if baseline_results:
            tbl.add_row(label, fused_val, base_val, speedup)
        else:
            tbl.add_row(label, fused_val)

    _row("PDFs", str(len(fused_results)))
    _row("Total pages", str(total_pages_f))
    _row(
        "Total time",
        f"{total_ms_f:.1f} ms",
        f"{total_ms_b:.1f} ms" if baseline_results else "",
        f"{total_ms_b / max(total_ms_f, 0.01):.1f}x" if baseline_results else "",
    )
    _row(
        "Avg / PDF",
        f"{avg_pdf_f:.1f} ms",
        f"{avg_pdf_b:.1f} ms" if baseline_results else "",
        f"{avg_pdf_b / max(avg_pdf_f, 0.01):.1f}x" if baseline_results else "",
    )
    _row(
        "Avg / page",
        f"{avg_page_f:.1f} ms",
        f"{avg_page_b:.1f} ms" if baseline_results else "",
        f"{avg_page_b / max(avg_page_f, 0.01):.1f}x" if baseline_results else "",
    )
    _row(
        "Pages / sec",
        f"{pps_f:.1f}",
        f"{pps_b:.1f}" if baseline_results else "",
        f"{pps_f / max(pps_b, 0.01):.1f}x" if baseline_results else "",
    )

    console.print()
    console.print(tbl)

    if baseline_results:
        speedup = total_ms_b / max(total_ms_f, 0.01)
        if speedup >= 2.0:
            console.print(f"\n[bold green]  >>> {speedup:.1f}x faster with the fused pipeline <<<[/bold green]\n")
        else:
            console.print(f"\n[bold yellow]  >>> {speedup:.1f}x  (fused vs baseline) <<<[/bold yellow]\n")


# ─── Per-stage head-to-head ──────────────────────────────────────────


def _print_per_stage_comparison(
    fused_results: list,
    baseline_results: list[dict],
    console: Console,
) -> None:
    """Aggregate timings per stage and show where the wins come from."""
    # Fused stage totals
    fused_stages: dict[str, float] = {}
    for r in fused_results:
        for key, ms in r.timings.items():
            if key == "total_ms":
                continue
            fused_stages[key] = fused_stages.get(key, 0.0) + ms

    # Baseline stage totals
    baseline_stages: dict[str, float] = {}
    for r in baseline_results:
        for m in r["metrics"]:
            name = m["stage"]
            ms = m["duration_sec"] * 1000
            baseline_stages[name] = baseline_stages.get(name, 0.0) + ms

    tbl = Table(
        title="Per-Stage Aggregate Timings",
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("Fused stage", style="green", min_width=24)
    tbl.add_column("ms", justify="right")
    tbl.add_column("│", style="dim")
    tbl.add_column("Baseline stage", style="yellow", min_width=24)
    tbl.add_column("ms", justify="right")

    fused_items = list(fused_stages.items())
    baseline_items = list(baseline_stages.items())
    max_rows = max(len(fused_items), len(baseline_items))
    for i in range(max_rows):
        fk, fv = fused_items[i] if i < len(fused_items) else ("", 0)
        bk, bv = baseline_items[i] if i < len(baseline_items) else ("", 0)
        fk_display = fk.replace("_ms", "").replace("_", " ") if fk else ""
        tbl.add_row(fk_display, f"{fv:.1f}" if fk else "", "│", bk, f"{bv:.1f}" if bk else "")

    console.print()
    console.print(tbl)


# ═════════════════════════════════════════════════════════════════════
# CLI commands
# ═════════════════════════════════════════════════════════════════════


@app.command("run")
def run_cmd(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory with PDFs"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="CUDA device"),
    warmup_runs: int = typer.Option(2, "--warmup-runs", help="Warmup iterations"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Also run baseline for comparison"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max PDFs to process"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Save results to JSON"),
) -> None:
    """Run the fused low-latency pipeline on a directory of PDFs."""
    console = Console()
    pdfs = _find_pdfs(input_dir, limit)
    console.print(f"\nFound [bold]{len(pdfs)}[/bold] PDFs in {input_dir}\n")

    # ── Fused pipeline ──
    fused_results = _run_fused(pdfs, device, warmup_runs, console)

    # ── Baseline (optional) ──
    baseline_results = None
    if compare:
        console.print()
        baseline_results = _run_baseline(pdfs, console)

    # ── Summary ──
    _print_summary(fused_results, baseline_results, console)
    if baseline_results:
        _print_per_stage_comparison(fused_results, baseline_results, console)

    # ── JSON output ──
    if output_json:
        _save_json(output_json, fused_results, baseline_results)
        console.print(f"\nResults saved to [bold]{output_json}[/bold]")


@app.command("baseline")
def baseline_cmd(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory with PDFs"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max PDFs to process"),
) -> None:
    """Run only the standard inprocess baseline for reference timing."""
    console = Console()
    pdfs = _find_pdfs(input_dir, limit)
    console.print(f"\nFound [bold]{len(pdfs)}[/bold] PDFs in {input_dir}\n")
    baseline_results = _run_baseline(pdfs, console)

    total_ms = sum(r["total_ms"] for r in baseline_results)
    total_pages = sum(r["num_pages"] for r in baseline_results)
    console.print(
        f"\n[bold]Total: {total_ms:.1f} ms  |  "
        f"Avg/PDF: {total_ms / max(len(baseline_results), 1):.1f} ms  |  "
        f"Avg/page: {total_ms / max(total_pages, 1):.1f} ms[/bold]\n"
    )


# ─── JSON export ─────────────────────────────────────────────────────


def _save_json(path: str, fused: list, baseline: list[dict] | None) -> None:
    data: dict = {
        "fused": [
            {
                "source": r.source_path,
                "num_pages": r.num_pages,
                "num_texts": len(r.all_texts),
                "embedding_dim": r.embedding_dim,
                "timings": dict(r.timings),
            }
            for r in fused
        ],
    }
    if baseline:
        data["baseline"] = baseline
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
