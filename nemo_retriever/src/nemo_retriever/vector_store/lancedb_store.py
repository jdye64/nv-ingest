# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

from nv_ingest_client.util.vdb.lancedb import LanceDB
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBConfig:
    """
    Minimal config for writing embeddings into LanceDB.

    This module is intentionally lightweight: it can be used by the text-embedding
    stage (`nemo_retriever.text_embed.stage`) and by the vector-store CLI (`nemo_retriever.vector_store.stage`).
    """

    uri: str = "lancedb"
    table_name: str = "nv-ingest"
    overwrite: bool = True

    # Optional index creation (recommended for recall/search runs).
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256

    hybrid: bool = False
    fts_language: str = "English"


def _read_text_embeddings_json_df(path: Path) -> pd.DataFrame:
    """
    Read a `*.text_embeddings.json` file emitted by `nemo_retriever.text_embed.stage`.

    Expected wrapper shape:
      {
        ...,
        "df_records": [ { "document_type": ..., "metadata": {...}, ... }, ... ],
        ...
      }
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        raise ValueError(f"Failed reading JSON {path}: {e}") from e

    if isinstance(obj, dict):
        recs = obj.get("df_records")
        if isinstance(recs, list):
            return pd.DataFrame([r for r in recs if isinstance(r, dict)])
        # Fall back to a single record.
        return pd.DataFrame([obj])

    if isinstance(obj, list):
        return pd.DataFrame([r for r in obj if isinstance(r, dict)])

    return pd.DataFrame([])


def _iter_text_embeddings_json_files(input_dir: Path, *, recursive: bool) -> List[Path]:
    """
    Return sorted list of `*.text_embeddings.json` files.

    The stage5 default naming is: `<input>.text_embeddings.json` (where `<input>` is
    typically a stage4 output filename).
    """
    if recursive:
        files = list(input_dir.rglob("*.text_embeddings.json"))
    else:
        files = list(input_dir.glob("*.text_embeddings.json"))
    return sorted([p for p in files if p.is_file()])


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_source_path_and_id(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract a stable source path/id from metadata.

    Prefers:
      - metadata.source_metadata.source_id
      - metadata.source_metadata.source_name
      - metadata.custom_content.path
    """
    source = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
    source_id = source.get("source_id") or ""
    source_name = source.get("source_name") or ""

    custom = meta.get("custom_content") if isinstance(meta.get("custom_content"), dict) else {}
    custom_path = custom.get("path") or custom.get("input_pdf") or custom.get("pdf_path") or ""

    path = _safe_str(custom_path or source_id or source_name)
    sid = _safe_str(source_id or path or source_name)
    return path, sid


def _extract_page_number(meta: Dict[str, Any]) -> int:
    cm = meta.get("content_metadata") if isinstance(meta.get("content_metadata"), dict) else {}
    page = cm.get("hierarchy", {}).get("page", -1)
    try:
        return int(page)
    except Exception:
        return -1


def _build_lancedb_rows_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Transform an embeddings-enriched primitives DataFrame into LanceDB rows.

    Rows include:
      - vector (embedding)
      - pdf_basename
      - page_number
      - pdf_page (basename_page)
      - source_id
      - path
    """
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue

        embedding = meta.get("embedding")
        if embedding is None:
            continue

        # Normalize embedding to list[float]
        if not isinstance(embedding, list):
            try:
                embedding = list(embedding)  # type: ignore[arg-type]
            except Exception:
                continue

        path, source_id = _extract_source_path_and_id(meta)
        page_number = _extract_page_number(meta)
        p = Path(path) if path else None
        filename = p.name if p is not None else ""
        pdf_basename = p.stem if p is not None else ""
        pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""

        if page_number == -1:
            logger.debug("Unable to determine page number for %s", path)

        out.append(
            {
                "vector": embedding,
                "pdf_page": pdf_page,
                "filename": filename,
                "pdf_basename": pdf_basename,
                "page_number": int(page_number),
                "source_id": source_id,
                "path": path,
            }
        )

    return out


def _infer_vector_dim(rows: Sequence[Dict[str, Any]]) -> int:
    for r in rows:
        v = r.get("vector")
        if isinstance(v, list) and v:
            return int(len(v))
    return 0


def create_lancedb_index(table: Any, *, cfg: LanceDBConfig, text_column: str = "text") -> None:
    """Create vector (IVF_HNSW_SQ) and optionally FTS indices on a LanceDB table."""
    try:
        table.create_index(
            index_type=cfg.index_type,
            metric=cfg.metric,
            num_partitions=int(cfg.num_partitions),
            num_sub_vectors=int(cfg.num_sub_vectors),
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")

    if cfg.hybrid:
        try:
            table.create_fts_index(text_column, language=cfg.fts_language)
        except Exception:
            logger.warning(
                "FTS index creation failed on column %r; continuing with vector-only search.",
                text_column,
                exc_info=True,
            )


def _write_rows_to_lancedb(rows: Sequence[Dict[str, Any]], *, cfg: LanceDBConfig) -> None:
    if not rows:
        logger.warning("No embeddings rows provided; nothing to write to LanceDB.")
        return

    dim = _infer_vector_dim(rows)
    if dim <= 0:
        raise ValueError("Failed to infer embedding dimension from rows.")

    try:
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LanceDB write requested but dependencies are missing. "
            "Install `lancedb` and `pyarrow` in this environment."
        ) from e

    db = lancedb.connect(uri=cfg.uri)

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
        ]
    )

    mode = "overwrite" if cfg.overwrite else "append"
    table = db.create_table(cfg.table_name, data=list(rows), schema=schema, mode=mode)

    if cfg.create_index:
        create_lancedb_index(table, cfg=cfg)


def write_embeddings_to_lancedb(df_with_embeddings: pd.DataFrame, *, cfg: LanceDBConfig) -> None:
    """
    Write embeddings found in `df_with_embeddings.metadata.embedding` to LanceDB.

    This is used programmatically by `nemo_retriever.text_embed.stage.embed_text_from_primitives_df(...)`.
    """
    rows = _build_lancedb_rows_from_df(df_with_embeddings)
    _write_rows_to_lancedb(rows, cfg=cfg)


def write_text_embeddings_dir_to_lancedb(
    input_dir: Path,
    *,
    cfg: LanceDBConfig,
    recursive: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read `*.text_embeddings.json` files from `input_dir` and upload their embeddings to LanceDB.
    """
    input_dir = Path(input_dir)
    files = _iter_text_embeddings_json_files(input_dir, recursive=bool(recursive))
    if limit is not None:
        files = files[: int(limit)]

    processed = 0
    skipped = 0
    failed = 0

    lancedb = LanceDB(uri=cfg.uri, table_name=cfg.table_name, overwrite=cfg.overwrite)

    results = []

    for p in files:
        df = _read_text_embeddings_json_df(p)
        rows = df.to_dict(orient="records")
        results.append(rows)

    if not results:
        logger.warning("No *.text_embeddings.json files found in %s; nothing to write.", input_dir)
        return {
            "input_dir": str(input_dir),
            "n_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
        }

    lancedb.run(results)

    return {
        "input_dir": str(input_dir),
        "n_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
    }


# ---------------------------------------------------------------------------
# Pipeline-level LanceDB helpers (used by batch_pipeline and similar scripts)
# ---------------------------------------------------------------------------


def _lancedb_connect():
    """Import lancedb lazily to avoid fork warnings during early process setup."""
    return import_module("lancedb")


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def ensure_lancedb_table(uri: str, table_name: str) -> None:
    """
    Ensure the local LanceDB URI exists and table can be opened.

    Creates an empty table with the expected schema if it does not exist yet.
    """
    Path(uri).mkdir(parents=True, exist_ok=True)

    db = _lancedb_connect().connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    import pyarrow as pa  # type: ignore

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2048)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    empty = pa.table(
        {
            k: []
            for k in [
                "vector",
                "pdf_page",
                "filename",
                "pdf_basename",
                "page_number",
                "source_id",
                "path",
                "text",
                "metadata",
                "source",
            ]
        },
        schema=schema,
    )
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def create_lancedb_indices(uri: str, table_name: str, *, hybrid: bool = False) -> None:
    """Create vector (and optional FTS) indices once after driver-side writes."""
    try:
        db = _lancedb_connect().connect(uri)
        table = db.open_table(table_name)
        n_rows = int(table.count_rows())
    except Exception as e:
        print(f"Warning: could not open LanceDB table for index creation: {e}")
        return

    if n_rows < 2:
        print("Skipping LanceDB index creation (not enough vectors).")
        return

    try:
        table.create_index(
            index_type="IVF_HNSW_SQ",
            metric="l2",
            num_partitions=16,
            num_sub_vectors=256,
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")
    except Exception as e:
        print(f"Warning: failed to create LanceDB vector index: {e}")

    if hybrid:
        try:
            table.create_fts_index("text", language="English")
        except Exception as e:
            print(f"Warning: failed to create LanceDB FTS index: {e}")


def stream_embeddings_to_driver_and_write_lancedb(
    *,
    ingestor: object,
    lancedb_uri: str,
    table_name: str,
    hybrid: bool = False,
    batch_size: int = 1024,
) -> int:
    """
    Stream embedded batches from Ray to the driver and write them locally to LanceDB.
    """
    from nemo_retriever.ingest_modes.inprocess import upload_embeddings_to_lancedb_inprocess  # noqa: PLC0415

    ds = getattr(ingestor, "_rd_dataset", None)
    if ds is None:
        raise RuntimeError("No Ray Dataset found on ingestor; call ingest() first.")

    first_write = True
    for batch_df in ds.iter_batches(batch_format="pandas", batch_size=int(batch_size)):
        if batch_df is None or batch_df.empty:
            continue
        upload_embeddings_to_lancedb_inprocess(
            batch_df,
            lancedb_uri=str(lancedb_uri),
            table_name=str(table_name),
            overwrite=bool(first_write),
            create_index=False,
            hybrid=bool(hybrid),
        )
        first_write = False

    create_lancedb_indices(str(lancedb_uri), str(table_name), hybrid=bool(hybrid))

    try:
        db = _lancedb_connect().connect(str(lancedb_uri))
        table = db.open_table(str(table_name))
        return int(table.count_rows())
    except Exception:
        return 0


def estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    """
    Estimate pages processed by counting unique (source_id, page_number) pairs.

    Falls back to table row count if page-level fields are unavailable.
    """
    try:
        db = _lancedb_connect().connect(uri)
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


def collect_detection_summary(uri: str, table_name: str) -> Optional[dict]:
    """
    Collect per-model detection totals deduped by (source_id, page_number).

    Counts are read from LanceDB row `metadata`, which is populated during batch
    ingestion by the Ray write stage.
    """
    try:
        db = _lancedb_connect().connect(uri)
        table = db.open_table(table_name)
        df = table.to_pandas()[["source_id", "page_number", "metadata"]]
    except Exception:
        return None

    per_page: dict[tuple[str, int], dict] = {}
    for row in df.itertuples(index=False):
        source_id = str(getattr(row, "source_id", "") or "")
        page_number = _to_int(getattr(row, "page_number", -1), default=-1)
        key = (source_id, page_number)

        raw_metadata = getattr(row, "metadata", None)
        meta: dict = {}
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}

        entry = per_page.setdefault(
            key,
            {
                "page_elements_total": 0,
                "ocr_table_total": 0,
                "ocr_chart_total": 0,
                "ocr_infographic_total": 0,
                "page_elements_by_label": defaultdict(int),
            },
        )

        entry["page_elements_total"] = max(
            entry["page_elements_total"], _to_int(meta.get("page_elements_v3_num_detections"))
        )
        entry["ocr_table_total"] = max(entry["ocr_table_total"], _to_int(meta.get("ocr_table_detections")))
        entry["ocr_chart_total"] = max(entry["ocr_chart_total"], _to_int(meta.get("ocr_chart_detections")))
        entry["ocr_infographic_total"] = max(
            entry["ocr_infographic_total"], _to_int(meta.get("ocr_infographic_detections"))
        )

        label_counts = meta.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                if isinstance(label, str):
                    entry["page_elements_by_label"][label] = max(entry["page_elements_by_label"][label], _to_int(count))

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    page_elements_total = ocr_table_total = ocr_chart_total = ocr_infographic_total = 0
    for page_entry in per_page.values():
        page_elements_total += int(page_entry["page_elements_total"])
        ocr_table_total += int(page_entry["ocr_table_total"])
        ocr_chart_total += int(page_entry["ocr_chart_total"])
        ocr_infographic_total += int(page_entry["ocr_infographic_total"])
        for label, count in page_entry["page_elements_by_label"].items():
            pe_by_label_totals[label] += int(count)

    return {
        "pages_seen": int(len(per_page)),
        "page_elements_v3_total_detections": int(page_elements_total),
        "page_elements_v3_counts_by_label": dict(sorted(pe_by_label_totals.items())),
        "ocr_table_total_detections": int(ocr_table_total),
        "ocr_chart_total_detections": int(ocr_chart_total),
        "ocr_infographic_total_detections": int(ocr_infographic_total),
    }


def print_detection_summary(summary: Optional[dict]) -> None:
    if summary is None:
        print("Detection summary: unavailable (could not read LanceDB metadata).")
        return
    print("\nDetection summary (deduped by source_id/page_number):")
    print(f"  Pages seen: {summary['pages_seen']}")
    print(f"  PageElements v3 total detections: {summary['page_elements_v3_total_detections']}")
    print(f"  OCR table detections: {summary['ocr_table_total_detections']}")
    print(f"  OCR chart detections: {summary['ocr_chart_total_detections']}")
    print(f"  OCR infographic detections: {summary['ocr_infographic_total_detections']}")
    print("  PageElements v3 counts by label:")
    by_label = summary.get("page_elements_v3_counts_by_label") or {}
    if not by_label:
        print("    (none)")
    else:
        for label, count in by_label.items():
            print(f"    {label}: {count}")


def write_detection_summary(path: Path, summary: Optional[dict]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = summary if summary is not None else {"error": "Detection summary unavailable."}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _extract_error_signals_from_value(value: object) -> list[str]:
    """Return human-readable error signals found in a value (supports nested dict/list)."""
    signals: list[str] = []

    if isinstance(value, dict):
        err = value.get("error")
        if err not in (None, "", {}, []):
            if isinstance(err, dict):
                parts = [p for p in (err.get("stage"), err.get("type"), err.get("message")) if p]
                signals.append("error=" + (" | ".join(str(p) for p in parts) if parts else str(err)))
            else:
                signals.append(f"error={err}")

        if value.get("errors") not in (None, "", {}, []):
            signals.append("errors field present")

        status = value.get("status")
        if isinstance(status, str) and status.lower() == "failed":
            signals.append("status=failed")

        if bool(value.get("cm_failed")):
            signals.append("cm_failed=true")

        if isinstance(value.get("failed"), bool) and value["failed"]:
            signals.append("failed=true")

        for nested in value.values():
            signals.extend(_extract_error_signals_from_value(nested))

    elif isinstance(value, list):
        for item in value:
            signals.extend(_extract_error_signals_from_value(item))

    elif isinstance(value, str):
        s = value.strip()
        if s:
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if parsed is not None:
                signals.extend(_extract_error_signals_from_value(parsed))
            elif any(kw in s.lower() for kw in ("traceback", "error", "exception")):
                signals.append("string contains error-like text")

    return signals


def _extract_error_details_from_value(value: object) -> list[dict[str, str]]:
    """Extract structured error details (stage/type/message) from nested values."""
    details: list[dict[str, str]] = []

    if isinstance(value, dict):
        err = value.get("error")
        if err not in (None, "", {}, []):
            if isinstance(err, dict):
                details.append(
                    {
                        "stage": str(err.get("stage") or "unknown"),
                        "type": str(err.get("type") or "unknown"),
                        "message": str(err.get("message") or ""),
                    }
                )
            else:
                details.append({"stage": "unknown", "type": "unknown", "message": str(err)})

        errors_field = value.get("errors")
        if isinstance(errors_field, list):
            for item in errors_field:
                if isinstance(item, dict):
                    details.append(
                        {
                            "stage": str(item.get("stage") or "unknown"),
                            "type": str(item.get("type") or "unknown"),
                            "message": str(item.get("message") or ""),
                        }
                    )

        for nested in value.values():
            details.extend(_extract_error_details_from_value(nested))

    elif isinstance(value, list):
        for item in value:
            details.extend(_extract_error_details_from_value(item))

    elif isinstance(value, str):
        s = value.strip()
        if s:
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if parsed is not None:
                details.extend(_extract_error_details_from_value(parsed))

    # De-duplicate while preserving order.
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for d in details:
        key = (d.get("stage", ""), d.get("type", ""), d.get("message", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped


def collect_ingest_row_errors(
    uri: str,
    table_name: str,
    *,
    preview_limit: int = 20,
) -> tuple[int, list[dict]]:
    """
    Scan LanceDB rows for error signals.

    Returns (total_error_rows, preview_rows).
    """
    try:
        db = _lancedb_connect().connect(uri)
        table = db.open_table(table_name)
        df = table.to_pandas()
    except Exception:
        return 0, []

    if df.empty:
        return 0, []

    total = 0
    preview: list[dict] = []
    inspect_cols = [c for c in ("source_id", "page_number", "metadata", "source", "text") if c in df.columns]

    for row in df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else {}
        signals: list[str] = []
        details: list[dict[str, str]] = []
        for col in inspect_cols:
            value = row_dict.get(col)
            signals.extend(_extract_error_signals_from_value(value))
            details.extend(_extract_error_details_from_value(value))

        deduped_signals = list(dict.fromkeys(signals))
        deduped_details: list[dict[str, str]] = []
        seen_details: set[tuple[str, str, str]] = set()
        for item in details:
            key = (item.get("stage", ""), item.get("type", ""), item.get("message", ""))
            if key not in seen_details:
                seen_details.add(key)
                deduped_details.append(item)

        if not deduped_signals:
            continue

        total += 1
        if len(preview) < preview_limit:
            preview.append(
                {
                    "source_id": row_dict.get("source_id"),
                    "page_number": row_dict.get("page_number"),
                    "signals": deduped_signals,
                    "errors": deduped_details,
                }
            )

    return total, preview
