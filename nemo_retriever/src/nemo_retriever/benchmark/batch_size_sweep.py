# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark inference batch sizes for NemotronPageElementsV3.

Loads PNG images from a directory, runs inference at each requested batch size,
and prints a comparison table so you can pick the optimal default.

Usage:
    python -m nemo_retriever.benchmark.batch_size_sweep \
        --input-dir /path/to/pngs \
        --batch-sizes 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_batch_sizes(raw: str) -> List[int]:
    sizes: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        if n <= 0:
            raise ValueError(f"Batch size must be > 0, got {n}")
        sizes.append(n)
    if not sizes:
        raise ValueError("At least one batch size is required")
    return sorted(set(sizes))


def _load_images(
    input_dir: Path,
    *,
    limit: int | None = None,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Load PNGs from *input_dir* as (HWC‑uint8 ndarray, (H, W)) pairs.

    When *limit* is set, only the first *limit* images (sorted by name) are loaded.
    """
    paths = sorted(input_dir.glob("*.png"))
    if not paths:
        sys.exit(f"No .png files found in {input_dir}")

    if limit is not None:
        paths = paths[:limit]

    logger.info("Loading %d images from %s", len(paths), input_dir)
    images: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    for p in tqdm(paths, desc="Loading images", unit="img"):
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            arr = np.array(rgb)  # (H, W, 3) uint8
            h, w = arr.shape[:2]
            images.append((arr, (h, w)))
    logger.info("Finished loading %d images", len(images))
    return images


def _run_batch_size(
    model: "NemotronPageElementsV3",  # noqa: F821
    images: List[Tuple[np.ndarray, Tuple[int, int]]],
    batch_size: int,
    warmup: bool = False,
) -> dict:
    """Run all images through the model at *batch_size* and return timing info."""
    n_images = len(images)
    n_batches = math.ceil(n_images / batch_size)

    pre_times: List[float] = []
    inf_times: List[float] = []

    if not warmup:
        logger.info("Starting batch_size=%d (%d batches, %d images)", batch_size, n_batches, n_images)

    pbar = tqdm(
        total=n_batches,
        desc=f"batch_size={batch_size}",
        unit="batch",
        disable=warmup,
    )
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_images)
        chunk = images[start:end]

        arrays = [arr for arr, _ in chunk]
        orig_shapes = [shape for _, shape in chunk]

        t0 = time.perf_counter()
        batch_tensor = model.preprocess_batch(arrays)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pre_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                _ = model.invoke(batch_tensor, orig_shapes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_times.append(time.perf_counter() - t0)
        pbar.update(1)
    pbar.close()

    total_pre = sum(pre_times)
    total_inf = sum(inf_times)

    if not warmup:
        logger.info(
            "Finished batch_size=%d — total=%.4fs (preprocess=%.4fs, inference=%.4fs)",
            batch_size,
            total_pre + total_inf,
            total_pre,
            total_inf,
        )

    return {
        "batch_size": batch_size,
        "n_images": n_images,
        "n_batches": n_batches,
        "total_preprocess_s": total_pre,
        "total_inference_s": total_inf,
        "total_s": total_pre + total_inf,
        "avg_inference_per_batch_s": statistics.mean(inf_times) if inf_times else 0.0,
        "images_per_second": n_images / (total_pre + total_inf) if (total_pre + total_inf) > 0 else 0.0,
    }


def _print_table(results: List[dict]) -> None:
    headers = [
        "batch_size",
        "n_batches",
        "preprocess(s)",
        "inference(s)",
        "total(s)",
        "avg_inf/batch(s)",
        "img/s",
    ]
    rows: List[List[str]] = []
    for r in results:
        rows.append(
            [
                str(r["batch_size"]),
                str(r["n_batches"]),
                f'{r["total_preprocess_s"]:.4f}',
                f'{r["total_inference_s"]:.4f}',
                f'{r["total_s"]:.4f}',
                f'{r["avg_inference_per_batch_s"]:.4f}',
                f'{r["images_per_second"]:.2f}',
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.rjust(w) for cell, w in zip(cells, col_widths))

    separator = "  ".join("-" * w for w in col_widths)

    print()
    print(_fmt_row(headers))
    print(separator)
    for row in rows:
        print(_fmt_row(row))

    best = min(results, key=lambda r: r["total_s"])
    print()
    print(
        f"Fastest: batch_size={best['batch_size']}  "
        f"total={best['total_s']:.4f}s  "
        f"img/s={best['images_per_second']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NemotronPageElementsV3 inference at various batch sizes.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .png images to use as input.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of batch sizes to benchmark (default: 1,2,4,8,16).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Run a single-image warmup pass before timing (default: true).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help="Skip the warmup pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load this many images for the sweep (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"--input-dir is not a directory: {input_dir}")

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    do_warmup = args.warmup and not args.no_warmup

    images = _load_images(input_dir, limit=args.limit)

    logger.info("Initializing NemotronPageElementsV3 model ...")
    from nemo_retriever.model.local import NemotronPageElementsV3

    model = NemotronPageElementsV3()

    if do_warmup:
        logger.info("Warming up (single image) ...")
        _run_batch_size(model, images[:1], batch_size=1, warmup=True)
        logger.info("Warmup complete.")

    logger.info("Benchmarking batch sizes %s over %d images", batch_sizes, len(images))

    results: List[dict] = []
    for bs in batch_sizes:
        r = _run_batch_size(model, images, batch_size=bs)
        results.append(r)

    _print_table(results)


if __name__ == "__main__":
    main()
