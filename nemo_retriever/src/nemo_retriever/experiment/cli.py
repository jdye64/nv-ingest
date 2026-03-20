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
    gpu0: str,
    gpu1: str,
    warmup_runs: int,
    render_workers: int,
    console: Console,
    *,
    score_threshold: float = 0.3,
    max_crops: int = 5,
    min_crop_px: int = 32,
    skip_ocr_if_text: bool = True,
    chunk_chars: int = 2000,
    chunk_overlap: int = 200,
    fast_embedder: bool = False,
    fast_embedder_model: str = "intfloat/e5-small-v2",
    _pipeline_out: list | None = None,
) -> list:
    from nemo_retriever.experiment.fused_pipeline import FusedPDFPipeline

    emb_tag = f"fast embedder ({fast_embedder_model})" if fast_embedder else "LlamaNemotron-1B"
    ocr_tag = "smart OCR routing" if skip_ocr_if_text else "full OCR"
    console.print(
        Panel(
            "[bold]Fused Pipeline — Dual GPU[/bold]\n"
            f"YOLOX+OCR on {gpu0} · embed on {gpu1} · "
            f"multiprocess render · {ocr_tag} · "
            f"chunking ({chunk_chars} chars) · {emb_tag}",
            style="green",
        )
    )

    pipeline = FusedPDFPipeline(
        gpu0=gpu0,
        gpu1=gpu1,
        render_workers=render_workers,
        score_threshold=score_threshold,
        max_crops_per_page=max_crops,
        min_crop_px=min_crop_px,
        skip_ocr_if_text=skip_ocr_if_text,
        chunk_target_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap,
        use_fast_embedder=fast_embedder,
        fast_embedder_model=fast_embedder_model,
    )

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

        name = pdf_path.name
        t = result.timings
        n_struct = sum(pr.num_structured_regions for pr in result.page_results)
        n_skipped = sum(1 for pr in result.page_results if pr.ocr_skipped)
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

        tbl.add_section()
        tbl.add_row("pages skipped OCR", str(n_skipped), style="dim")
        tbl.add_row("OCR crops sent", str(result.ocr_stats.get("crops_sent", 0)), style="dim")
        tbl.add_row("OCR crops filtered", str(result.ocr_stats.get("crops_skipped", 0)), style="dim")
        tbl.add_row("chunks embedded", str(n_text), style="dim")
        tbl.add_row("embedding shape", str(emb_shape), style="dim")
        console.print(tbl)
        console.print()

    if _pipeline_out is not None:
        _pipeline_out.append(pipeline)

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
    fused_stages: dict[str, float] = {}
    for r in fused_results:
        for key, ms in r.timings.items():
            if key == "total_ms":
                continue
            fused_stages[key] = fused_stages.get(key, 0.0) + ms

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
    gpu0: str = typer.Option("cuda:0", "--gpu0", help="GPU for YOLOX + OCR"),
    gpu1: str = typer.Option("cuda:1", "--gpu1", help="GPU for embedder"),
    warmup_runs: int = typer.Option(2, "--warmup-runs", help="Warmup iterations"),
    render_workers: int = typer.Option(0, "--render-workers", help="Render process count (0=auto)"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Also run baseline for comparison"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max PDFs to process"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Save results to JSON"),
    # --- OCR filtering ---
    score_threshold: float = typer.Option(0.3, "--score-threshold", help="Min detection confidence for OCR"),
    max_crops: int = typer.Option(5, "--max-crops", help="Max OCR crops per page (top-K by score)"),
    min_crop_px: int = typer.Option(32, "--min-crop-px", help="Min crop dimension in pixels"),
    no_skip_ocr: bool = typer.Option(False, "--no-skip-ocr", help="Disable smart OCR skipping for text-rich pages"),
    # --- Chunking ---
    chunk_chars: int = typer.Option(2000, "--chunk-chars", help="Target chunk size in characters"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Overlap between chunks in characters"),
    # --- Embedder ---
    fast_embedder: bool = typer.Option(False, "--fast-embedder", help="Use a small fast embedder instead of 1B"),
    fast_embedder_model: str = typer.Option("intfloat/e5-small-v2", "--fast-embedder-model", help="HF model ID for fast embedder"),
    # --- LanceDB + Recall ---
    query_csv: Optional[str] = typer.Option(None, "--query-csv", "-q", help="Query GT CSV for recall evaluation"),
    lancedb_uri: str = typer.Option("/tmp/experiment_lancedb", "--lancedb-uri", help="LanceDB database URI"),
    lancedb_table: str = typer.Option("experiment", "--lancedb-table", help="LanceDB table name"),
) -> None:
    """Run the fused low-latency pipeline on a directory of PDFs."""
    console = Console()
    pdfs = _find_pdfs(input_dir, limit)
    console.print(f"\nFound [bold]{len(pdfs)}[/bold] PDFs in {input_dir}\n")

    pipeline_holder: list = []
    fused_results = _run_fused(
        pdfs, gpu0, gpu1, warmup_runs, render_workers, console,
        score_threshold=score_threshold,
        max_crops=max_crops,
        min_crop_px=min_crop_px,
        skip_ocr_if_text=not no_skip_ocr,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        fast_embedder=fast_embedder,
        fast_embedder_model=fast_embedder_model,
        _pipeline_out=pipeline_holder,
    )

    baseline_results = None
    if compare:
        console.print()
        baseline_results = _run_baseline(pdfs, console)

    _print_summary(fused_results, baseline_results, console)
    if baseline_results:
        _print_per_stage_comparison(fused_results, baseline_results, console)

    # ── LanceDB write ──
    console.print("\n[bold]Writing embeddings to LanceDB …[/bold]")
    _write_lancedb(fused_results, lancedb_uri, lancedb_table, console)

    # ── Recall evaluation (reuses the already-loaded embedder on gpu1) ──
    if query_csv:
        embedder = pipeline_holder[0].embedder if pipeline_holder else None
        if embedder is None:
            if fast_embedder:
                from nemo_retriever.experiment.fused_pipeline import FastEmbedder
                embedder = FastEmbedder(model_id=fast_embedder_model, device=gpu1)
            else:
                from nemo_retriever.model import create_local_embedder
                embedder = create_local_embedder(device=gpu1)

        _run_recall(
            lancedb_uri, lancedb_table, query_csv, embedder, console,
        )

    # Clean up render process pool
    if pipeline_holder:
        pipeline_holder[0].shutdown()

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


def _write_lancedb(
    fused_results: list,
    db_uri: str,
    table_name: str,
    console: Console,
) -> None:
    """Write all chunk embeddings to a LanceDB table."""
    import lancedb
    import pyarrow as pa

    rows = []
    for r in fused_results:
        if r.embeddings is None:
            continue
        pdf_stem = Path(r.source_path).stem
        emb_list = r.embeddings.cpu().float().tolist()
        for idx, (chunk, emb) in enumerate(zip(r.chunks, emb_list)):
            for pn in chunk.page_numbers:
                rows.append({
                    "vector": emb,
                    "text": chunk.text[:500],
                    "pdf_page": f"{pdf_stem}_{pn}",
                    "filename": Path(r.source_path).name,
                    "pdf_basename": pdf_stem,
                    "page_number": pn,
                    "source_id": r.source_path,
                    "path": r.source_path,
                    "chunk_index": idx,
                })

    if not rows:
        console.print("[yellow]No embeddings to write to LanceDB[/yellow]")
        return

    dim = len(rows[0]["vector"])
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("text", pa.string()),
        pa.field("pdf_page", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("pdf_basename", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("source_id", pa.string()),
        pa.field("path", pa.string()),
        pa.field("chunk_index", pa.int32()),
    ])

    db = lancedb.connect(db_uri)
    tbl = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")
    console.print(
        f"  Wrote [bold]{len(rows)}[/bold] rows to LanceDB "
        f"([dim]{db_uri}[/dim] / [bold]{table_name}[/bold]) — "
        f"vector dim={dim}"
    )
    return tbl


def _run_recall(
    db_uri: str,
    table_name: str,
    query_csv: str,
    embedder: object,
    console: Console,
    ks: tuple[int, ...] = (1, 5, 10),
    top_k: int = 10,
) -> dict[str, float]:
    """Embed queries, search LanceDB, and compute Recall@K."""
    import lancedb
    import pandas as pd

    console.print(f"\n[bold]Running recall evaluation …[/bold]")

    df = pd.read_csv(query_csv)
    queries = df["query"].astype(str).tolist()

    # Build golden keys
    if "pdf_page" in df.columns:
        gold = df["pdf_page"].astype(str).tolist()
    else:
        df["pdf"] = df["pdf"].astype(str).str.replace(".pdf", "", regex=False)
        df["page"] = df["page"].astype(str)
        gold = (df["pdf"] + "_" + df["page"]).tolist()

    console.print(f"  Queries: {len(queries)}, Top-K: {top_k}")

    # Embed queries — use "query: " prefix for e5 models, standard for others
    import torch
    t0 = time.perf_counter()
    prefixed = [f"query: {q}" for q in queries]
    query_embs = embedder.embed(prefixed, batch_size=64)
    torch.cuda.synchronize()
    embed_s = time.perf_counter() - t0
    console.print(f"  Query embedding: {embed_s:.2f}s ({len(queries)} queries)")

    # Search LanceDB
    db = lancedb.connect(db_uri)
    tbl = db.open_table(table_name)

    t0 = time.perf_counter()
    retrieved_keys: list[list[str]] = []
    for i in range(len(queries)):
        vec = query_embs[i].cpu().float().tolist()
        hits = tbl.search(vec).limit(top_k).to_list()
        keys = [h["pdf_page"] for h in hits if h.get("pdf_page")]
        retrieved_keys.append(keys)
    search_s = time.perf_counter() - t0
    console.print(f"  LanceDB search: {search_s:.2f}s")

    # Compute recall
    metrics: dict[str, float] = {}
    for k in ks:
        hits = 0
        for g, rets in zip(gold, retrieved_keys):
            if g in rets[:k]:
                hits += 1
        metrics[f"recall@{k}"] = hits / max(1, len(gold))

    # Print results
    tbl_out = Table(title="Recall Evaluation", show_header=True, header_style="bold")
    tbl_out.add_column("Metric", style="cyan")
    tbl_out.add_column("Value", justify="right", style="bold green")
    for name, val in metrics.items():
        tbl_out.add_row(name, f"{val:.4f}  ({val * 100:.1f}%)")
    tbl_out.add_section()
    tbl_out.add_row("queries", str(len(queries)), style="dim")
    tbl_out.add_row("query embed time", f"{embed_s:.2f}s", style="dim")
    tbl_out.add_row("search time", f"{search_s:.2f}s", style="dim")
    console.print()
    console.print(tbl_out)

    return metrics


def _save_json(path: str, fused: list, baseline: list[dict] | None) -> None:
    data: dict = {
        "fused": [
            {
                "source": r.source_path,
                "num_pages": r.num_pages,
                "num_texts": len(r.all_texts),
                "embedding_dim": r.embedding_dim,
                "ocr_stats": r.ocr_stats,
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
