# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m retriever.examples.batch_pipeline <input-dir>
"""

import subprocess
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import ray
import typer
from retriever import create_ingestor
from retriever.params import EmbedParams
from retriever.params import ExtractParams
from retriever.params import IngestExecuteParams
from retriever.params import IngestorCreateParams
from retriever.params import TextChunkParams
from retriever.params import VdbUploadParams
from retriever.recall.core import (
    RecallConfig,
    _ensure_lancedb_table,
    _gold_to_doc_page,
    _hit_key_and_distance,
    _is_hit_at_k,
    _lancedb,
    retrieve_and_score,
)

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"
VALID_INPUT_EXTENSIONS = [".pdf", ".txt", ".html", ".docx", ".pptx"]


def _validate_system_resources(
    ray_address: Optional[str],
    start_ray: bool,
    pdf_extract_workers: int,
    pdf_extract_num_cpus: float,
    pdf_extract_batch_size: int,
    pdf_split_batch_size: int,
    page_elements_batch_size: int,
    ocr_workers: int,
    page_elements_workers: int,
    ocr_batch_size: int,
    page_elements_cpus_per_actor: float,
    ocr_cpus_per_actor: float,
    embed_workers: int,
    embed_batch_size: int,
    embed_cpus_per_actor: float,
    gpu_page_elements: float,
    gpu_ocr: float,
    gpu_embed: float,
    page_elements_invoke_url: Optional[str],
    ocr_invoke_url: Optional[str],
    embed_invoke_url: Optional[str],
) -> dict[str, object]:
    int_fields = {
        "pdf_extract_workers": pdf_extract_workers,
        "pdf_extract_batch_size": pdf_extract_batch_size,
        "pdf_split_batch_size": pdf_split_batch_size,
        "page_elements_batch_size": page_elements_batch_size,
        "ocr_workers": ocr_workers,
        "page_elements_workers": page_elements_workers,
        "ocr_batch_size": ocr_batch_size,
        "embed_workers": embed_workers,
        "embed_batch_size": embed_batch_size,
    }
    for name, value in int_fields.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")

    positive_float_fields = {
        "pdf_extract_num_cpus": pdf_extract_num_cpus,
        "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
        "ocr_cpus_per_actor": ocr_cpus_per_actor,
        "embed_cpus_per_actor": embed_cpus_per_actor,
    }
    for name, value in positive_float_fields.items():
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be > 0.0, got {value}.")

    non_negative_float_fields = {
        "gpu_page_elements": gpu_page_elements,
        "gpu_ocr": gpu_ocr,
        "gpu_embed": gpu_embed,
    }
    for name, value in non_negative_float_fields.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0.0, got {value}.")

    if start_ray and ray_address:
        print("[WARN] --start-ray is set; ignoring --ray-address and connecting to local head node.")

    # Remote endpoints don't need local model GPUs for their stage.
    if page_elements_invoke_url and float(gpu_page_elements) != 0.0:
        print(
            "[WARN] --page-elements-invoke-url is set; forcing --gpu-page-elements from "
            f"{float(gpu_page_elements):.3f} to 0.0"
        )
        gpu_page_elements = 0.0

    if ocr_invoke_url and float(gpu_ocr) != 0.0:
        print("[WARN] --ocr-invoke-url is set; forcing --gpu-ocr from " f"{float(gpu_ocr):.3f} to 0.0")
        gpu_ocr = 0.0

    if embed_invoke_url and float(gpu_embed) != 0.0:
        print("[WARN] --embed-invoke-url is set; forcing --gpu-embed from " f"{float(gpu_embed):.3f} to 0.0")
        gpu_embed = 0.0

    return {
        "ray_address": ray_address,
        "start_ray": bool(start_ray),
        "pdf_extract_workers": int(pdf_extract_workers),
        "pdf_extract_num_cpus": float(pdf_extract_num_cpus),
        "pdf_extract_batch_size": int(pdf_extract_batch_size),
        "pdf_split_batch_size": int(pdf_split_batch_size),
        "page_elements_batch_size": int(page_elements_batch_size),
        "ocr_workers": int(ocr_workers),
        "page_elements_workers": int(page_elements_workers),
        "ocr_batch_size": int(ocr_batch_size),
        "page_elements_cpus_per_actor": float(page_elements_cpus_per_actor),
        "ocr_cpus_per_actor": float(ocr_cpus_per_actor),
        "embed_workers": int(embed_workers),
        "embed_batch_size": int(embed_batch_size),
        "embed_cpus_per_actor": float(embed_cpus_per_actor),
        "gpu_page_elements": float(gpu_page_elements),
        "gpu_ocr": float(gpu_ocr),
        "gpu_embed": float(gpu_embed),
        "page_elements_invoke_url": page_elements_invoke_url,
        "ocr_invoke_url": ocr_invoke_url,
        "embed_invoke_url": embed_invoke_url,
    }


def _print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print("Pages/sec: unavailable (could not estimate processed pages). " f"Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only; excludes Ray startup and recall): {pps:.2f}")


def _estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    try:
        db = _lancedb().connect(uri)
        table = db.open_table(table_name)
    except Exception:
        return None

    try:
        df = table.to_pandas()[["source_id", "page_number"]]
        return int(df.dropna(subset=["source_id", "page_number"]).drop_duplicates().shape[0])
    except Exception:
        try:
            return int(table.count_rows())
        except Exception:
            return None


def _collect_detection_summary(uri: str, table_name: str) -> dict:
    summary = {
        "pages_seen": 0,
        "page_elements_v3_total_detections": 0,
        "page_elements_v3_counts_by_label": {},
        "ocr_table_total_detections": 0,
        "ocr_chart_total_detections": 0,
        "ocr_infographic_total_detections": 0,
    }
    try:
        db = _lancedb().connect(uri)
        table = db.open_table(table_name)
        df = table.to_pandas()
    except Exception:
        return summary

    per_page: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        meta = row_dict.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        source = row_dict.get("source")
        if isinstance(source, str):
            try:
                source = json.loads(source)
            except Exception:
                source = {}
        if not isinstance(source, dict):
            source = {}

        source_id = str(
            row_dict.get("source_id")
            or source.get("source_id")
            or meta.get("source_path")
            or row_dict.get("path")
            or ""
        )
        try:
            page_number = int(meta.get("page_number") or row_dict.get("page_number") or -1)
        except (TypeError, ValueError):
            page_number = -1

        key = (source_id, page_number)
        entry = per_page.setdefault(
            key,
            {
                "pe": 0,
                "ocr_table": 0,
                "ocr_chart": 0,
                "ocr_infographic": 0,
                "pe_by_label": defaultdict(int),
            },
        )

        try:
            pe = int(
                meta.get("page_elements_v3_num_detections") or row_dict.get("page_elements_v3_num_detections") or 0
            )
        except (TypeError, ValueError):
            pe = 0
        entry["pe"] = max(entry["pe"], pe)

        for field, meta_key in [
            ("ocr_table", "ocr_table_detections"),
            ("ocr_chart", "ocr_chart_detections"),
            ("ocr_infographic", "ocr_infographic_detections"),
        ]:
            try:
                val = int(meta.get(meta_key, 0) or 0)
            except (TypeError, ValueError):
                val = 0
            entry[field] = max(entry[field], val)

        label_counts = meta.get("page_elements_v3_counts_by_label") or row_dict.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                try:
                    c = int(count or 0)
                except (TypeError, ValueError):
                    c = 0
                label_name = str(label)
                entry["pe_by_label"][label_name] = max(entry["pe_by_label"][label_name], c)

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    for entry in per_page.values():
        summary["page_elements_v3_total_detections"] += int(entry["pe"])
        summary["ocr_table_total_detections"] += int(entry["ocr_table"])
        summary["ocr_chart_total_detections"] += int(entry["ocr_chart"])
        summary["ocr_infographic_total_detections"] += int(entry["ocr_infographic"])
        for label, count in entry["pe_by_label"].items():
            pe_by_label_totals[label] += int(count)

    summary["pages_seen"] = len(per_page)
    summary["page_elements_v3_counts_by_label"] = dict(sorted(pe_by_label_totals.items()))
    return summary


def _print_detection_summary(summary: dict) -> None:
    print("\nDetection summary (deduped by source/page_number):")
    print(f"  Pages seen: {summary.get('pages_seen', 0)}")
    print(f"  PageElements v3 total detections: {summary.get('page_elements_v3_total_detections', 0)}")
    print(f"  OCR table detections: {summary.get('ocr_table_total_detections', 0)}")
    print(f"  OCR chart detections: {summary.get('ocr_chart_total_detections', 0)}")
    print(f"  OCR infographic detections: {summary.get('ocr_infographic_total_detections', 0)}")
    print("  PageElements v3 counts by label:")
    by_label = summary.get("page_elements_v3_counts_by_label", {})
    if not by_label:
        print("    (none)")
        return
    for label, count in by_label.items():
        print(f"    {label}: {count}")


def _write_detection_summary(out_path: Path, summary: dict) -> None:
    out_file = Path(out_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help=("Directory containing files to ingest. Supported extensions: " f"{', '.join(VALID_INPUT_EXTENSIONS)}"),
        path_type=Path,
        exists=True,
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    start_ray: bool = typer.Option(
        False,
        "--start-ray",
        help=(
            "Start a Ray head node (ray start --head) and connect to it. "
            "Dashboard at http://127.0.0.1:8265. Ignores --ray-address."
        ),
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help=(
            "Path to query CSV for recall evaluation. Default: bo767_query_gt.csv "
            "(current directory). Recall is skipped if the file does not exist."
        ),
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help=(
            "Do not print per-query retrieval details (query, gold, hits). "
            "Only the missed-gold summary and recall metrics are printed."
        ),
    ),
    pdf_extract_workers: int = typer.Option(
        12,
        "--pdf-extract-workers",
        min=1,
        help="Number of CPU workers for PDF extraction stage.",
    ),
    pdf_extract_num_cpus: float = typer.Option(
        2.0,
        "--pdf-extract-num-cpus",
        min=0.1,
        help="CPUs reserved per PDF extraction task.",
    ),
    pdf_extract_batch_size: int = typer.Option(
        4,
        "--pdf-extract-batch-size",
        min=1,
        help="Batch size for PDF extraction stage.",
    ),
    pdf_split_batch_size: int = typer.Option(
        1,
        "--pdf-split-batch-size",
        min=1,
        help="Batch size for PDF split stage.",
    ),
    page_elements_batch_size: int = typer.Option(
        24,
        "--page-elements-batch-size",
        min=1,
        help="Ray Data batch size for page-elements stage.",
    ),
    ocr_workers: int = typer.Option(
        1,
        "--ocr-workers",
        min=1,
        help="Actor count for OCR stage.",
    ),
    page_elements_workers: int = typer.Option(
        1,
        "--page-elements-workers",
        min=1,
        help="Actor count for page-elements stage.",
    ),
    ocr_batch_size: int = typer.Option(
        16,
        "--ocr-batch-size",
        min=1,
        help="Ray Data batch size for OCR stage.",
    ),
    page_elements_cpus_per_actor: float = typer.Option(
        1.0,
        "--page-elements-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per page-elements actor.",
    ),
    ocr_cpus_per_actor: float = typer.Option(
        1.0,
        "--ocr-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per OCR actor.",
    ),
    embed_workers: int = typer.Option(
        1,
        "--embed-workers",
        min=1,
        help="Actor count for embedding stage.",
    ),
    embed_batch_size: int = typer.Option(
        256,
        "--embed-batch-size",
        min=1,
        help="Ray Data batch size for embedding stage.",
    ),
    embed_cpus_per_actor: float = typer.Option(
        1.0,
        "--embed-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per embedding actor.",
    ),
    gpu_page_elements: float = typer.Option(
        0.5,
        "--gpu-page-elements",
        min=0.0,
        help="GPUs reserved per page-elements actor.",
    ),
    gpu_ocr: float = typer.Option(
        1.0,
        "--gpu-ocr",
        min=0.0,
        help="GPUs reserved per OCR actor.",
    ),
    gpu_embed: float = typer.Option(
        0.5,
        "--gpu-embed",
        min=0.0,
        help="GPUs reserved per embedding actor.",
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Optional remote endpoint URL for page-elements model inference.",
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None,
        "--ocr-invoke-url",
        help="Optional remote endpoint URL for OCR model inference.",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Optional remote endpoint URL for embedding model inference.",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None,
        "--runtime-metrics-dir",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        help="Optional directory where Ray runtime metrics are written per run.",
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(
        None,
        "--runtime-metrics-prefix",
        help="Optional filename prefix for per-run metrics artifacts.",
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        help="Optional path to write detection summary JSON after ingest.",
    ),
    print_actor_metrics_json: bool = typer.Option(
        False,
        "--print-actor-metrics-json",
        help="Print end-of-run actor metrics report JSON emitted by the MetricsActor.",
    ),
) -> None:
    try:
        system_resources = _validate_system_resources(
            ray_address=ray_address,
            start_ray=start_ray,
            pdf_extract_workers=pdf_extract_workers,
            pdf_extract_num_cpus=pdf_extract_num_cpus,
            pdf_extract_batch_size=pdf_extract_batch_size,
            pdf_split_batch_size=pdf_split_batch_size,
            page_elements_batch_size=page_elements_batch_size,
            ocr_workers=ocr_workers,
            page_elements_workers=page_elements_workers,
            ocr_batch_size=ocr_batch_size,
            page_elements_cpus_per_actor=page_elements_cpus_per_actor,
            ocr_cpus_per_actor=ocr_cpus_per_actor,
            embed_workers=embed_workers,
            embed_batch_size=embed_batch_size,
            embed_cpus_per_actor=embed_cpus_per_actor,
            gpu_page_elements=gpu_page_elements,
            gpu_ocr=gpu_ocr,
            gpu_embed=gpu_embed,
            page_elements_invoke_url=page_elements_invoke_url,
            ocr_invoke_url=ocr_invoke_url,
            embed_invoke_url=embed_invoke_url,
        )
        ray_address = system_resources["ray_address"]
        start_ray = bool(system_resources["start_ray"])
        pdf_extract_workers = int(system_resources["pdf_extract_workers"])
        pdf_extract_num_cpus = float(system_resources["pdf_extract_num_cpus"])
        pdf_extract_batch_size = int(system_resources["pdf_extract_batch_size"])
        pdf_split_batch_size = int(system_resources["pdf_split_batch_size"])
        page_elements_batch_size = int(system_resources["page_elements_batch_size"])
        ocr_workers = int(system_resources["ocr_workers"])
        page_elements_workers = int(system_resources["page_elements_workers"])
        ocr_batch_size = int(system_resources["ocr_batch_size"])
        page_elements_cpus_per_actor = float(system_resources["page_elements_cpus_per_actor"])
        ocr_cpus_per_actor = float(system_resources["ocr_cpus_per_actor"])
        embed_workers = int(system_resources["embed_workers"])
        embed_batch_size = int(system_resources["embed_batch_size"])
        embed_cpus_per_actor = float(system_resources["embed_cpus_per_actor"])
        gpu_page_elements = float(system_resources["gpu_page_elements"])
        gpu_ocr = float(system_resources["gpu_ocr"])
        gpu_embed = float(system_resources["gpu_embed"])
        page_elements_invoke_url = system_resources["page_elements_invoke_url"]
        ocr_invoke_url = system_resources["ocr_invoke_url"]
        embed_invoke_url = system_resources["embed_invoke_url"]

        # Use an absolute path so driver and Ray actors resolve the same LanceDB URI.
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        # Resolve Ray: start a head node, connect to given address, or run in-process
        if start_ray:
            subprocess.run(["ray", "start", "--head"], check=True)
            ray_address = "auto"

        input_dir = Path(input_dir)
        matched_files = sorted({p for ext in VALID_INPUT_EXTENSIONS for p in input_dir.glob(f"*{ext}") if p.is_file()})
        if not matched_files:
            raise ValueError(
                "No supported input files found in input_dir. " f"Expected one of: {', '.join(VALID_INPUT_EXTENSIONS)}"
            )
        matched_extensions = {p.suffix.lower() for p in matched_files}

        if matched_extensions == {".txt"}:
            input_globs = [str(input_dir / "*.txt")]
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address),
            )
            ingestor = (
                ingestor.files(input_globs)
                .extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(EmbedParams(model_name=str(embed_model_name), embed_invoke_url=embed_invoke_url))
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
            recall_ext = ".txt"
        elif matched_extensions == {".html"}:
            input_globs = [str(input_dir / "*.html")]
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address),
            )
            ingestor = (
                ingestor.files(input_globs)
                .extract_html(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(EmbedParams(model_name=str(embed_model_name), embed_invoke_url=embed_invoke_url))
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
            recall_ext = ".html"
        elif matched_extensions.issubset({".docx", ".pptx"}):
            # DOCX/PPTX: same pipeline as PDF; DocToPdfConversionActor converts before split.
            input_globs = [str(input_dir / f"*{ext}") for ext in sorted(matched_extensions)]
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address),
            )
            ingestor = (
                ingestor.files(input_globs)
                .extract(
                    ExtractParams(
                        extract_text=True,
                        extract_tables=True,
                        extract_charts=True,
                        extract_infographics=False,
                        page_elements_invoke_url=page_elements_invoke_url,
                        ocr_invoke_url=ocr_invoke_url,
                        batch_tuning={
                            "debug_run_id": str(runtime_metrics_prefix or "unknown"),
                            "pdf_extract_workers": int(pdf_extract_workers),
                            "pdf_extract_num_cpus": float(pdf_extract_num_cpus),
                            "pdf_split_batch_size": int(pdf_split_batch_size),
                            "pdf_extract_batch_size": int(pdf_extract_batch_size),
                            "page_elements_batch_size": int(page_elements_batch_size),
                            "page_elements_workers": int(page_elements_workers),
                            "detect_workers": int(ocr_workers),
                            "detect_batch_size": int(ocr_batch_size),
                            "page_elements_cpus_per_actor": float(page_elements_cpus_per_actor),
                            "ocr_cpus_per_actor": float(ocr_cpus_per_actor),
                            "gpu_page_elements": float(gpu_page_elements),
                            "gpu_ocr": float(gpu_ocr),
                            "gpu_embed": float(gpu_embed),
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        batch_tuning={
                            "embed_workers": int(embed_workers),
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
            recall_ext = ".docx"
        elif matched_extensions == {".pdf"}:
            input_globs = [str(input_dir / "*.pdf")]
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address),
            )
            ingestor = (
                ingestor.files(input_globs)
                .extract(
                    ExtractParams(
                        extract_text=True,
                        extract_tables=True,
                        extract_charts=True,
                        extract_infographics=False,
                        page_elements_invoke_url=page_elements_invoke_url,
                        ocr_invoke_url=ocr_invoke_url,
                        batch_tuning={
                            "debug_run_id": str(runtime_metrics_prefix or "unknown"),
                            "pdf_extract_workers": int(pdf_extract_workers),
                            "pdf_extract_num_cpus": float(pdf_extract_num_cpus),
                            "pdf_split_batch_size": int(pdf_split_batch_size),
                            "pdf_extract_batch_size": int(pdf_extract_batch_size),
                            "page_elements_batch_size": int(page_elements_batch_size),
                            "page_elements_workers": int(page_elements_workers),
                            "detect_workers": int(ocr_workers),
                            "detect_batch_size": int(ocr_batch_size),
                            "page_elements_cpus_per_actor": float(page_elements_cpus_per_actor),
                            "ocr_cpus_per_actor": float(ocr_cpus_per_actor),
                            "gpu_page_elements": float(gpu_page_elements),
                            "gpu_ocr": float(gpu_ocr),
                            "gpu_embed": float(gpu_embed),
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        batch_tuning={
                            "embed_workers": int(embed_workers),
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
            recall_ext = ".pdf"
        else:
            raise ValueError(
                "Found mixed or unsupported file extensions in input_dir: "
                f"{', '.join(sorted(matched_extensions))}. "
                "Please provide a directory with one supported input family "
                "(only .pdf, only .txt, only .html, or only .docx/.pptx)."
            )

        print("Running extraction...")
        ingest_start = time.perf_counter()
        ingestor.ingest(
            params=IngestExecuteParams(
                runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
                runtime_metrics_prefix=runtime_metrics_prefix,
            )
        )
        if print_actor_metrics_json:
            actor_report = getattr(ingestor, "_last_actor_metrics_report", None)
            if isinstance(actor_report, dict):
                print("\nActor metrics report (JSON):")
                print(json.dumps(actor_report, indent=2, sort_keys=True))
            else:
                print("\nActor metrics report unavailable.")
        ingest_elapsed_s = time.perf_counter() - ingest_start
        processed_pages = _estimate_processed_pages(lancedb_uri, LANCEDB_TABLE)
        detection_summary = _collect_detection_summary(lancedb_uri, LANCEDB_TABLE)
        print("Extraction complete.")
        _print_detection_summary(detection_summary)
        if detection_summary_file is not None:
            _write_detection_summary(detection_summary_file, detection_summary)
            print(f"Wrote detection summary JSON to {Path(detection_summary_file).expanduser().resolve()}")

        ray.shutdown()

        # ---------------------------------------------------------------------------
        # Recall calculation (optional)
        # ---------------------------------------------------------------------------
        query_csv = Path(query_csv)
        if not query_csv.exists():
            print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return

        db = _lancedb().connect(lancedb_uri)
        table = None
        open_err: Optional[Exception] = None
        for _ in range(3):
            try:
                table = db.open_table(LANCEDB_TABLE)
                open_err = None
                break
            except Exception as e:
                open_err = e
                # Create table if missing, then retry open.
                _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)
                time.sleep(2)
        if table is None:
            raise RuntimeError(
                f"Recall stage requires LanceDB table {LANCEDB_TABLE!r} at {lancedb_uri!r}, " f"but it was not found."
            ) from open_err
        try:
            if int(table.count_rows()) == 0:
                print(f"LanceDB table {LANCEDB_TABLE!r} exists but is empty; skipping recall evaluation.")
                _print_pages_per_second(processed_pages, ingest_elapsed_s)
                return
        except Exception:
            pass

        # Resolve the HF model ID for recall query embedding so aliases
        # (e.g. "nemo_retriever_v1") map to the correct model.
        from retriever.model import resolve_embed_model

        _recall_model = resolve_embed_model(str(embed_model_name))

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model=_recall_model,
            top_k=10,
            ks=(1, 5, 10),
            hybrid=hybrid,
        )

        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

        if not no_recall_details:
            print("\nPer-query retrieval details:")
        missed_gold: list[tuple[str, str]] = []
        ext = recall_ext
        for i, (q, g, hits) in enumerate(
            zip(
                _df_query["query"].astype(str).tolist(),
                _gold,
                _raw_hits,
            )
        ):
            doc, page = _gold_to_doc_page(g)

            scored_hits: list[tuple[str, float | None]] = []
            for h in hits:
                key, dist = _hit_key_and_distance(h)
                if key:
                    scored_hits.append((key, dist))

            top_keys = [k for (k, _d) in scored_hits]
            hit = _is_hit_at_k(g, top_keys, cfg.top_k)

            if not no_recall_details:
                print(f"\nQuery {i}: {q}")
                print(f"  Gold: {g}  (file: {doc}{ext}, page: {page})")
                print(f"  Hit@{cfg.top_k}: {hit}")
                print("  Top hits:")
                if not scored_hits:
                    print("    (no hits)")
                else:
                    for rank, (key, dist) in enumerate(scored_hits[: int(cfg.top_k)], start=1):
                        if dist is None:
                            print(f"    {rank:02d}. {key}")
                        else:
                            print(f"    {rank:02d}. {key}  distance={dist:.6f}")

            if not hit:
                missed_gold.append((f"{doc}{ext}", str(page)))

        missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
        print("\nMissed gold (unique doc/page):")
        if not missed_unique:
            print("  (none)")
        else:
            for doc_page, page in missed_unique:
                print(f"  {doc_page} page {page}")
        print(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")

        print("\nRecall metrics (matching retriever.recall.core):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
    finally:
        pass


if __name__ == "__main__":
    app()
