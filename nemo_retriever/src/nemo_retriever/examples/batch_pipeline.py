# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.batch_pipeline <input-dir>
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

import ray
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.recall.core import (
    RecallConfig,
    gold_to_doc_page,
    hit_key_and_distance,
    is_hit_at_k,
    retrieve_and_score,
)
from nemo_retriever.vector_store.lancedb_store import (
    ensure_lancedb_table,
    estimate_processed_pages,
    stream_embeddings_to_driver_and_write_lancedb,
)

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


class _TeeStream:
    """Write stream output to terminal and optional log file."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(getattr(self._primary, "fileno")())

    def writable(self) -> bool:
        return bool(getattr(self._primary, "writable", lambda: True)())

    @property
    def encoding(self) -> str:
        return str(getattr(self._primary, "encoding", "utf-8"))


def _configure_logging(log_file: Optional[Path]) -> tuple[Optional[TextIO], TextIO, TextIO]:
    """Configure root logging; optionally tee stdout/stderr into one file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if log_file is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return None, original_stdout, original_stderr

    target = Path(log_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = open(target, "a", encoding="utf-8", buffering=1)

    # Tee stdout/stderr so print(), tracebacks, and Ray driver-forwarded logs
    # all land in the same place while still showing on the console.
    sys.stdout = _TeeStream(sys.__stdout__, fh)
    sys.stderr = _TeeStream(sys.__stderr__, fh)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("Writing combined pipeline logs to %s", str(target))
    return fh, original_stdout, original_stderr


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


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
        exists=True,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', or 'doc'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice).",  # noqa: E501
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
    # fmt: off
    nemotron_parse_workers: float = typer.Option(
        0.0,
        "--nemotron-parse-workers",
        min=0.0,
        help=(
            "Actor count for Nemotron Parse stage "
            "(enables parse-only mode when > 0.0 with parse GPU/batch-size)."
        ),  # noqa: E501
    ),
    # fmt: on
    gpu_nemotron_parse: float = typer.Option(
        0.0,
        "--gpu-nemotron-parse",
        min=0.0,
        help="GPUs reserved per Nemotron Parse actor.",
    ),
    nemotron_parse_batch_size: float = typer.Option(
        0.0,
        "--nemotron-parse-batch-size",
        min=0.0,
        help="Ray Data batch size for Nemotron Parse stage "
        "(enables parse-only mode when > 0.0 with parse workers/GPU).",
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
    embed_modality: str = typer.Option(
        "text",
        "--embed-modality",
        help="Default embedding modality for all element types: "
        "'text', 'image', or 'text_image' ('image_text' is also accepted).",
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None,
        "--text-elements-modality",
        help="Embedding modality override for page-text rows. Falls back to --embed-modality.",
    ),
    structured_elements_modality: Optional[str] = typer.Option(
        None,
        "--structured-elements-modality",
        help="Embedding modality override for table/chart/infographic rows. Falls back to --embed-modality.",
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
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        path_type=Path,
        dir_okay=False,
        help="Optional file to collect all pipeline + Ray driver logs for this run.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver (recommended with --log-file).",
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Optional JSON file path to write end-of-run detection counts summary.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
) -> None:
    log_handle, original_stdout, original_stderr = _configure_logging(log_file)
    try:
        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        # Use an absolute path so driver and Ray actors resolve the same LanceDB URI.
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

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

        # Resolve Ray: start a head node, connect to given address, or run in-process
        if start_ray:
            subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
            ray_address = "auto"

        input_dir = Path(input_dir)
        if input_type == "txt":
            glob_pattern = str(input_dir / "*.txt")
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
            )
            ingestor = (
                ingestor.files(glob_pattern)
                .extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                    )
                )
            )
        elif input_type == "html":
            glob_pattern = str(input_dir / "*.html")
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
            )
            ingestor = (
                ingestor.files(glob_pattern)
                .extract_html(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                    )
                )
            )
        elif input_type == "doc":
            # DOCX/PPTX: same pipeline as PDF; DocToPdfConversionActor converts before split.
            doc_globs = [str(input_dir / "*.docx"), str(input_dir / "*.pptx")]
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
            )
            ingestor = (
                ingestor.files(doc_globs)
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
                            "nemotron_parse_workers": float(nemotron_parse_workers),
                            "gpu_nemotron_parse": float(gpu_nemotron_parse),
                            "nemotron_parse_batch_size": float(nemotron_parse_batch_size),
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        batch_tuning={
                            "embed_workers": int(embed_workers),
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
            )
        else:
            pdf_glob = str(input_dir / "*.pdf")
            ingestor = create_ingestor(
                run_mode="batch",
                params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
            )
            ingestor = (
                ingestor.files(pdf_glob)
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
                            "nemotron_parse_workers": float(nemotron_parse_workers),
                            "gpu_nemotron_parse": float(gpu_nemotron_parse),
                            "nemotron_parse_batch_size": float(nemotron_parse_batch_size),
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        batch_tuning={
                            "embed_workers": int(embed_workers),
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
            )

        print("Running extraction...")
        ingest_start = time.perf_counter()
        ingest_result = ingestor.ingest(
            params=IngestExecuteParams(
                return_failures=True,
                runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
                runtime_metrics_prefix=runtime_metrics_prefix,
            )
        )
        if isinstance(ingest_result, tuple) and len(ingest_result) >= 2:
            num_pages, failures = ingest_result
        else:
            num_pages = int(ingest_result) if isinstance(ingest_result, int) else 0
            failures = []
        ingest_elapsed_s = time.perf_counter() - ingest_start
        print(f"Ingest complete: {num_pages} pages. Streaming embeddings to driver for local LanceDB writes...")
        write_start = time.perf_counter()
        written_rows = stream_embeddings_to_driver_and_write_lancedb(
            ingestor=ingestor,
            lancedb_uri=lancedb_uri,
            table_name=LANCEDB_TABLE,
            hybrid=hybrid,
            batch_size=1024,
        )
        write_elapsed_s = time.perf_counter() - write_start
        print(f"Driver-side LanceDB write complete: {written_rows} rows in {write_elapsed_s:.1f}s")
        processed_pages = estimate_processed_pages(lancedb_uri, LANCEDB_TABLE)
        print("Extraction complete.")

        if failures:
            print("\nDetected row-level errors returned by ingest().")
            error_preview = failures[:20]
            print(f"First {len(error_preview)} error rows:")
            for i, item in enumerate(error_preview, start=1):
                print(
                    f"  {i:02d}. source_id={item.get('source_id')!r}, "
                    f"path={item.get('path')!r}, "
                    f"page_number={item.get('page_number')!r}, "
                    f"errors={item.get('errors')}"
                )
            print(f"Total error rows returned: {len(failures)}")
            print("Skipping recall because ingestion errors were detected.")
            ray.shutdown()
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return
        else:
            print("No errors detected in ingestion output.")

        ray.shutdown()

        # ---------------------------------------------------------------------------
        # Recall calculation (optional)
        # ---------------------------------------------------------------------------
        query_csv = Path(query_csv)
        if not query_csv.exists():
            print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return

        import lancedb  # noqa: PLC0415

        db = lancedb.connect(lancedb_uri)
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
                ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)
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
        unique_basenames = table.to_pandas()["pdf_basename"].unique()
        print(f"Unique basenames: {unique_basenames}")

        # Resolve the HF model ID for recall query embedding so aliases
        # (e.g. "nemo_retriever_v1") map to the correct model.
        from nemo_retriever.model import resolve_embed_model

        _recall_model = resolve_embed_model(str(embed_model_name))

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model=_recall_model,
            embedding_http_endpoint=embed_invoke_url,
            top_k=10,
            ks=(1, 5, 10),
            hybrid=hybrid,
        )

        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

        if not no_recall_details:
            print("\nPer-query retrieval details:")
        missed_gold: list[tuple[str, str]] = []
        ext = (
            ".html"
            if input_type == "html"
            else (".txt" if input_type == "txt" else (".docx" if input_type == "doc" else ".pdf"))
        )
        for i, (q, g, hits) in enumerate(
            zip(
                _df_query["query"].astype(str).tolist(),
                _gold,
                _raw_hits,
            )
        ):
            doc, page = gold_to_doc_page(g)

            scored_hits: list[tuple[str, float | None]] = []
            for h in hits:
                key, dist = hit_key_and_distance(h)
                if key:
                    scored_hits.append((key, dist))

            top_keys = [k for (k, _d) in scored_hits]
            hit = is_hit_at_k(g, top_keys, cfg.top_k)

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

        print("\nRecall metrics (matching nemo_retriever.recall.core):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        processed_pages = 54730
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
    finally:
        # Restore real stdio before closing the mirror file so exception hooks
        # and late flushes never write to a closed stream wrapper.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            try:
                log_handle.flush()
            finally:
                log_handle.close()


if __name__ == "__main__":
    app()
