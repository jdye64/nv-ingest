#!/usr/bin/env python3
"""
Run PDFExtractionCPUActor in isolation and persist every page image
and its row-level metadata to disk.

Usage
-----
    python run_pdf_extraction.py /path/to/doc.pdf [more pdfs …]

    # Override the output root (default: $OUTPUT_PATH or cwd)
    OUTPUT_PATH=/tmp/results python run_pdf_extraction.py docs/*.pdf

Output layout
-------------
    $OUTPUT_PATH/pdf_extraction/
        <doc_stem>_<page_num>.jpeg      # rendered page image
        <doc_stem>_<page_num>.json      # row metadata (no raw pixel blobs)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.pdf.split import pdf_path_to_pages_df
from nemo_retriever.pdf.extract import PDFExtractionCPUActor


def _serialisable_page_image(val: Any) -> Any:
    """Strip raw bytes from page_image but keep shape / size info."""
    if isinstance(val, dict):
        return {
            "orig_shape_hw": val.get("orig_shape_hw"),
            "jpeg_bytes_len": len(val.get("jpeg_bytes", b"")),
        }
    return None


def _serialisable_images(val: Any) -> Any:
    """Keep image metadata but drop bulky base64 payloads."""
    if not isinstance(val, list):
        return val
    out: List[Dict[str, Any]] = []
    for img in val:
        if isinstance(img, dict):
            out.append({k: v for k, v in img.items() if k != "image_b64"})
        else:
            out.append(img)
    return out


def _row_to_metadata(row: pd.Series, columns: pd.Index) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for col in columns:
        val = row[col]
        if col == "page_image":
            record[col] = _serialisable_page_image(val)
        elif col == "images":
            record[col] = _serialisable_images(val)
        else:
            record[col] = val
    return record


def process_pdf(pdf_path: Path, out_dir: Path, actor: PDFExtractionCPUActor) -> int:
    """Extract pages from *pdf_path*, write images + metadata, return page count."""
    doc_stem = pdf_path.stem

    pages_df = pdf_path_to_pages_df(str(pdf_path))
    if pages_df.empty:
        print(f"  WARNING: no pages extracted from {pdf_path}", file=sys.stderr)
        return 0

    result = actor(pages_df)

    if isinstance(result, list):
        result = pd.DataFrame(result)
    if not isinstance(result, pd.DataFrame):
        print(f"  ERROR: unexpected result type {type(result)}", file=sys.stderr)
        return 0

    saved = 0
    for _, row in result.iterrows():
        page_num = int(row.get("page_number", 0))
        page_image = row.get("page_image")

        # -- page image (JPEG produced by _render_page_to_base64) --
        if isinstance(page_image, dict):
            jpeg_bytes = page_image.get("jpeg_bytes")
            if jpeg_bytes:
                img_file = out_dir / f"{doc_stem}_{page_num}.jpeg"
                img_file.write_bytes(jpeg_bytes)
                print(f"    {img_file.name}")
                saved += 1

        # -- embedded image crops (base64 PNG from pdfium) --
        images = row.get("images")
        if isinstance(images, list):
            for idx, img_obj in enumerate(images):
                if not isinstance(img_obj, dict):
                    continue
                b64 = img_obj.get("image_b64")
                if not b64:
                    continue
                crop_file = out_dir / f"{doc_stem}_{page_num}_crop_{idx}.png"
                crop_file.write_bytes(base64.b64decode(b64))
                print(f"    {crop_file.name}")

        # -- row metadata as JSON --
        meta = _row_to_metadata(row, result.columns)
        json_file = out_dir / f"{doc_stem}_{page_num}.json"
        json_file.write_text(json.dumps(meta, indent=2, default=str))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PDFExtractionCPUActor and save page images + metadata.",
    )
    parser.add_argument("input_pdfs", nargs="+", help="PDF file path(s)")
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", "."),
        help="Base output dir (default: $OUTPUT_PATH or cwd)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--extract-text", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_path) / "pdf_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    actor = PDFExtractionCPUActor(
        extract_text=args.extract_text,
        extract_images=True,
        dpi=args.dpi,
        image_format="jpeg",
    )

    total = 0
    for raw_path in args.input_pdfs:
        pdf = Path(raw_path).resolve()
        if not pdf.is_file():
            print(f"Skipping {raw_path} (not a file)", file=sys.stderr)
            continue
        print(f"Processing {pdf.name} …")
        total += process_pdf(pdf, out_dir, actor)

    print(f"\nDone — {total} page images saved to {out_dir}")


if __name__ == "__main__":
    main()
