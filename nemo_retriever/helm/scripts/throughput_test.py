#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput smoke-test for a deployed nemo-retriever "service" instance.

Walks an input directory, hands every file to ``create_ingestor(run_mode="service")``,
consumes the resulting async event stream from ``aingest_stream()``, and reports
pages-per-second once every page has finished processing.

Live counters of every event type are printed to stderr so transport problems
(stream errors, dropped uploads, server overflow) are immediately visible.

Examples::

    python throughput_test.py --input-dir ./pdfs
    python throughput_test.py --input-dir ./pdfs --url http://10.86.5.28:30670/ \\
        --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

from nemo_retriever import create_ingestor

DEFAULT_URL = "http://10.86.5.28:30670/"
PROGRESS_INTERVAL_S = 2.0


def _iter_input_files(input_dir: Path) -> list[str]:
    """Return every regular file under ``input_dir`` (recursive), sorted."""
    return [str(p) for p in sorted(input_dir.rglob("*")) if p.is_file()]


def _print_progress(counts: Counter, files_total: int, elapsed: float) -> None:
    pages = counts.get("page_result", 0)
    failed = counts.get("page_failed", 0)
    started = counts.get("job_started", 0)
    completed = counts.get("job_complete", 0)
    pps = pages / elapsed if elapsed > 0 else 0.0
    print(
        f"[{elapsed:6.1f}s] jobs {completed}/{started}/{files_total} " f"pages={pages} failed={failed} PPS={pps:6.2f}",
        file=sys.stderr,
        flush=True,
    )


async def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        return 2

    files = _iter_input_files(input_dir)
    if not files:
        print(f"No files found under {input_dir}", file=sys.stderr)
        return 1

    base_url = args.url.rstrip("/")
    print(f"Target:           {base_url}", file=sys.stderr)
    print(f"Files:            {len(files)} under {input_dir}", file=sys.stderr)
    print(f"Max concurrency:  {args.max_concurrency}", file=sys.stderr)

    ingestor = create_ingestor(
        run_mode="service",
        base_url=base_url,
        documents=files,
        api_key=args.api_key or None,
        max_concurrency=args.max_concurrency,
    )

    counts: Counter = Counter()
    stream_errors: list[str] = []
    overflow_seen = False

    t0 = time.monotonic()
    last_progress = t0

    async for event in ingestor.aingest_stream():
        etype = event.get("event", "message")
        counts[etype] += 1

        if etype == "stream_error":
            err = event.get("error", "<unknown>")
            stream_errors.append(err)
            print(f"!! stream_error: {err}", file=sys.stderr, flush=True)
        elif etype == "stream_overflow":
            overflow_seen = True
            print(
                "!! stream_overflow: server lost events (in-memory bus full); "
                "throughput numbers will be a lower bound",
                file=sys.stderr,
                flush=True,
            )
        elif etype == "page_failed":
            print(
                f"!! page_failed: {event.get('source_file')} p{event.get('page_number')} " f"-- {event.get('error')}",
                file=sys.stderr,
                flush=True,
            )

        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            _print_progress(counts, len(files), now - t0)
            last_progress = now

    elapsed = time.monotonic() - t0
    pages = counts.get("page_result", 0)
    failed = counts.get("page_failed", 0)
    pps = pages / elapsed if elapsed > 0 else 0.0

    print()
    print(f"Files:    {len(files)}")
    print(f"Pages:    {pages}  (failed {failed})")
    print(f"Elapsed:  {elapsed:.2f}s")
    print(f"PPS:      {pps:.2f} pages/sec")
    print()
    print("Event counts:")
    for name in sorted(counts):
        print(f"  {name:<20} {counts[name]}")
    if stream_errors:
        print()
        print(f"Stream errors ({len(stream_errors)}):")
        for err in stream_errors:
            print(f"  - {err}")
    if overflow_seen:
        print()
        print("NOTE: server emitted stream_overflow; some events were dropped.")

    return 0 if (pages > 0 and not stream_errors) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory of files to ingest (walked recursively).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Service base URL (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional bearer token forwarded to the service.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help=(
            "Maximum concurrent page uploads (default: 4).  Lower this "
            "(e.g. 2) if you see httpx.ReadError from a NodePort path."
        ),
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
