# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick CLI report for per-page ingest metrics.

Examples
--------
uv run python -m nemo_retriever.examples.per_page_metrics_report \
  --per-page-metrics ./run.per_page_metrics.parquet

uv run python -m nemo_retriever.examples.per_page_metrics_report \
  --runtime-summary ./run.runtime.summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer

app = typer.Typer(help="Inspect per-page metrics and print latency/count summaries.")


def _find_latest_runtime_summary(metrics_dir: Path) -> Optional[Path]:
    matches = sorted(metrics_dir.glob("*.runtime.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_per_page_metrics_path(
    per_page_metrics: Optional[Path],
    runtime_summary: Optional[Path],
    metrics_dir: Optional[Path],
) -> Path:
    if per_page_metrics is not None:
        p = per_page_metrics.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Per-page metrics file not found: {p}")
        return p

    summary_path: Optional[Path] = None
    if runtime_summary is not None:
        summary_path = runtime_summary.expanduser().resolve()
    elif metrics_dir is not None:
        summary_path = _find_latest_runtime_summary(metrics_dir.expanduser().resolve())

    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(
            "Could not resolve per-page metrics path. Provide --per-page-metrics, "
            "--runtime-summary, or --metrics-dir with a *.runtime.summary.json file."
        )

    payload = _load_json(summary_path)
    per_page_path = payload.get("per_page_metrics_path")
    if not per_page_path:
        raise RuntimeError(f"'per_page_metrics_path' is missing in runtime summary: {summary_path}")
    p = Path(str(per_page_path)).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Per-page metrics path in summary does not exist: {p}")
    return p


def _read_per_page_metrics(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported per-page metrics format: {path.suffix}")


def _percentile_summary(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "count": float(len(s)),
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def _print_percentiles(df: pd.DataFrame) -> None:
    latency_cols = [
        "extract_total_actor_ms",
        "extract_text_ms",
        "extract_render_ms",
        "pe_avg_per_row_ms",
        "ocr_avg_per_row_ms",
        "parse_avg_per_row_ms",
        "embed_avg_per_row_ms",
    ]
    available = [c for c in latency_cols if c in df.columns]
    if not available:
        typer.echo("\nNo latency columns found in per-page metrics table.")
        return

    typer.echo("\nLatency percentiles (ms):")
    typer.echo("column,count,mean,p50,p90,p95,p99,max")
    for col in available:
        stats = _percentile_summary(df[col])
        if not stats:
            continue
        typer.echo(
            f"{col},{int(stats['count'])},{stats['mean']:.2f},{stats['p50']:.2f},"
            f"{stats['p90']:.2f},{stats['p95']:.2f},{stats['p99']:.2f},{stats['max']:.2f}"
        )


def _print_slowest_pages(df: pd.DataFrame, *, primary_col: str, top_n: int) -> None:
    if primary_col not in df.columns:
        typer.echo(f"\nPrimary sort column not found: {primary_col}")
        return
    view_cols = [
        c for c in ["path", "page_number", primary_col, "pe_detections_total", "ocr_table_count"] if c in df.columns
    ]
    if not view_cols:
        return
    slow = df.sort_values(primary_col, ascending=False).head(int(top_n))
    typer.echo(f"\nTop {int(top_n)} slowest pages by {primary_col}:")
    typer.echo(slow[view_cols].to_string(index=False))


def _print_top_docs(df: pd.DataFrame, *, primary_col: str, top_n: int) -> None:
    if primary_col not in df.columns or "path" not in df.columns:
        return
    g = (
        df.groupby("path", as_index=False)[primary_col]
        .agg(page_count="count", mean_ms="mean", p95_ms=lambda x: x.quantile(0.95), max_ms="max")
        .sort_values("mean_ms", ascending=False)
        .head(int(top_n))
    )
    typer.echo(f"\nTop {int(top_n)} documents by mean {primary_col}:")
    typer.echo(g.to_string(index=False))


@app.command("run")
def run(
    per_page_metrics: Optional[Path] = typer.Option(
        None, "--per-page-metrics", help="Path to *.per_page_metrics.parquet or *.jsonl"
    ),
    runtime_summary: Optional[Path] = typer.Option(
        None, "--runtime-summary", help="Path to *.runtime.summary.json (uses per_page_metrics_path field)"
    ),
    metrics_dir: Optional[Path] = typer.Option(
        None, "--metrics-dir", help="Directory containing runtime summary files; latest one is used"
    ),
    primary_latency_col: str = typer.Option(
        "extract_total_actor_ms", "--primary-latency-col", help="Column used to rank slowest pages/docs"
    ),
    top_n_pages: int = typer.Option(20, "--top-n-pages", min=1, help="Number of slow pages to print"),
    top_n_docs: int = typer.Option(20, "--top-n-docs", min=1, help="Number of slow docs to print"),
) -> None:
    path = _resolve_per_page_metrics_path(
        per_page_metrics=per_page_metrics,
        runtime_summary=runtime_summary,
        metrics_dir=metrics_dir,
    )
    df = _read_per_page_metrics(path)
    if df.empty:
        typer.echo(f"Per-page metrics table is empty: {path}")
        return

    typer.echo(f"Loaded per-page metrics: {path}")
    typer.echo(f"Rows: {len(df)}")
    if "path" in df.columns:
        typer.echo(f"Documents: {df['path'].nunique()}")
    if {"path", "page_number"}.issubset(df.columns):
        typer.echo(f"Unique pages: {df[['path', 'page_number']].drop_duplicates().shape[0]}")
    if "has_error" in df.columns:
        try:
            typer.echo(f"Rows with errors: {int(df['has_error'].fillna(False).astype(bool).sum())}")
        except Exception:
            pass

    _print_percentiles(df)
    _print_slowest_pages(df, primary_col=str(primary_latency_col), top_n=int(top_n_pages))
    _print_top_docs(df, primary_col=str(primary_latency_col), top_n=int(top_n_docs))


if __name__ == "__main__":
    app()
