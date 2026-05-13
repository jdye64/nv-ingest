#!/usr/bin/env python3
"""Inspect a LanceDB table written by `retriever ingest` / `pipeline run`.

Prints row count, schema, modality breakdown (text / table / chart / image
counts pulled from `metadata.type`), per-source row counts, and up to N
sample rows. Use this to confirm an ingest actually populated the table.

Examples
--------

Inspect the default table (`lancedb/nv-ingest.lance`):

    python inspect_lancedb.py lancedb nv-ingest

Inspect a custom table, show 5 sample rows, no truncation of text:

    python inspect_lancedb.py ./my-lancedb my-corpus --samples 5 --no-truncate

Filter sample rows to a specific source PDF basename:

    python inspect_lancedb.py lancedb nv-ingest --source 1102434
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _truncate(value: Any, limit: int) -> str:
    s = str(value)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uri", help="LanceDB URI (directory path).")
    parser.add_argument("table", help="LanceDB table name.")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to print (default: 3).")
    parser.add_argument(
        "--truncate",
        dest="truncate",
        type=int,
        default=120,
        help="Truncate sample text to this many chars (default: 120; pass --no-truncate to disable).",
    )
    parser.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_const",
        const=0,
        help="Do not truncate sample text.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Filter sample rows to a specific pdf_basename / source substring.",
    )
    args = parser.parse_args(argv)

    try:
        import lancedb
    except ImportError:
        print("ERROR: lancedb not installed; `pip install lancedb` in the same env as retriever.", file=sys.stderr)
        return 2

    uri_path = Path(args.uri).expanduser()
    if not uri_path.exists():
        print(f"ERROR: LanceDB URI does not exist: {uri_path}", file=sys.stderr)
        return 1

    try:
        db = lancedb.connect(str(uri_path))
        table = db.open_table(args.table)
    except Exception as exc:
        print(f"ERROR: opening LanceDB table {args.table!r} at {uri_path}: {exc}", file=sys.stderr)
        try:
            available = ", ".join(db.table_names())  # type: ignore[name-defined]
            print(f"Available tables: {available or '(none)'}", file=sys.stderr)
        except Exception:
            pass
        return 1

    n_rows = len(table)
    print(f"== LanceDB table: {uri_path}/{args.table}.lance ==")
    print(f"rows: {n_rows}")
    print()

    print("Schema:")
    schema = table.schema
    for field in schema:
        print(f"  {field.name:24s} {field.type}")
    print()

    if n_rows == 0:
        print("Table is empty — no ingest has populated it, or ingest wrote elsewhere.")
        return 0

    df = table.to_pandas()

    if "metadata" in df.columns:
        modality_counter: Counter[str] = Counter()
        page_numbers: list[int] = []
        for raw in df["metadata"].tolist():
            meta = _parse_metadata(raw)
            modality = str(meta.get("type", "unknown"))
            modality_counter[modality] += 1
            pn = meta.get("page_number")
            if isinstance(pn, int):
                page_numbers.append(pn)
        print("Modality breakdown (metadata.type):")
        for modality, count in modality_counter.most_common():
            print(f"  {modality:12s} {count}")
        if page_numbers:
            print()
            print(f"page_number: min={min(page_numbers)}  max={max(page_numbers)}  unique={len(set(page_numbers))}")
        print()

    if "source" in df.columns:
        source_basenames = []
        for raw in df["source"].tolist():
            src = _parse_metadata(raw) if not isinstance(raw, (dict, type(None))) else raw
            if isinstance(src, dict):
                source_id = src.get("source_id") or src.get("source_name") or ""
                source_basenames.append(Path(str(source_id)).stem if source_id else "")
            else:
                source_basenames.append(str(raw) if raw else "")
        source_counts = Counter(s for s in source_basenames if s)
        if source_counts:
            print(f"Per-source row counts (top 10 of {len(source_counts)} sources):")
            for source, count in source_counts.most_common(10):
                print(f"  {count:6d}  {source}")
            print()

    print(f"Sample rows ({min(args.samples, n_rows)} of {n_rows}):")
    sample_df = df
    if args.source:
        if "source" in df.columns:
            mask = df["source"].astype(str).str.contains(args.source, na=False)
            sample_df = df[mask]
            print(f"  (filtered to source containing {args.source!r}: {len(sample_df)} rows)")
            if sample_df.empty:
                return 0

    for idx, (_, row) in enumerate(sample_df.head(args.samples).iterrows()):
        print(f"\n  --- sample {idx} ---")
        for col in ("text", "source", "metadata", "page_number", "pdf_basename", "pdf_page", "source_id", "_distance"):
            if col not in row.index:
                continue
            value = row[col]
            if value is None:
                continue
            if args.truncate > 0:
                value_str = _truncate(value, args.truncate)
            else:
                value_str = str(value)
            print(f"  {col:14s} {value_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
