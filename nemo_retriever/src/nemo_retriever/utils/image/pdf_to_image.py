# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert every page of every PDF in a directory to PNG or JPEG images."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from tqdm import tqdm

console = Console()
app = typer.Typer(help="Convert PDF pages to PNG or JPEG images.")


class ImageFormat(str, Enum):
    png = "png"
    jpeg = "jpeg"


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    *,
    image_format: ImageFormat = ImageFormat.png,
    dpi: int = 200,
) -> list[Path]:
    """Render every page of *pdf_path* to *output_dir* and return the written paths."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    stem = pdf_path.stem
    ext = "jpg" if image_format == ImageFormat.jpeg else "png"
    pil_format = "JPEG" if image_format == ImageFormat.jpeg else "PNG"
    scale = max(float(dpi) / 72.0, 0.01)

    written: list[Path] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()

        out_path = output_dir / f"{stem}_page_{page_idx + 1:04d}.{ext}"
        save_kwargs: dict = {}
        if image_format == ImageFormat.jpeg:
            save_kwargs["quality"] = 95
        pil_img.save(out_path, format=pil_format, **save_kwargs)
        written.append(out_path)
        page.close()

    doc.close()
    return written


def _convert_one_pdf(
    pdf_path: Path,
    output_dir: Path,
    image_format: ImageFormat,
    dpi: int,
) -> list[Path]:
    """Top-level helper so each call is picklable for ProcessPoolExecutor."""
    return convert_pdf_to_images(pdf_path, output_dir, image_format=image_format, dpi=dpi)


def convert_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    image_format: ImageFormat = ImageFormat.png,
    dpi: int = 200,
    limit: Optional[int] = None,
    workers: int = 16,
) -> list[Path]:
    """Convert all PDFs in *input_dir* to images in *output_dir*."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        console.print(f"[yellow]No .pdf files found in {input_dir}[/yellow]")
        return []

    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    effective_workers = min(workers, len(pdf_paths))

    all_written: list[Path] = []
    if effective_workers <= 1:
        for pdf_path in tqdm(pdf_paths, desc="Converting PDFs", unit="pdf"):
            try:
                all_written.extend(convert_pdf_to_images(pdf_path, output_dir, image_format=image_format, dpi=dpi))
            except Exception as exc:
                console.print(f"[red]Error[/red] converting {pdf_path.name}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            future_to_path = {pool.submit(_convert_one_pdf, p, output_dir, image_format, dpi): p for p in pdf_paths}
            with tqdm(total=len(pdf_paths), desc="Converting PDFs", unit="pdf") as pbar:
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    try:
                        all_written.extend(future.result())
                    except Exception as exc:
                        console.print(f"[red]Error[/red] converting {pdf_path.name}: {exc}")
                    pbar.update(1)

    return all_written


@app.command("convert")
def convert(
    input_dir: Path = typer.Option(
        ..., "--input-dir", exists=True, file_okay=False, dir_okay=True, help="Directory containing .pdf files."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", file_okay=False, dir_okay=True, help="Directory to write output images."
    ),
    image_format: ImageFormat = typer.Option(ImageFormat.png, "--format", help="Output image format (png or jpeg)."),
    dpi: int = typer.Option(200, "--dpi", min=72, help="Render resolution in DPI."),
    workers: int = typer.Option(16, "--workers", min=1, help="Number of PDFs to convert in parallel."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max number of PDFs to convert."),
) -> None:
    """Convert all PDF files in --input-dir to images written to --output-dir."""
    written = convert_dir(
        input_dir,
        output_dir,
        image_format=image_format,
        dpi=dpi,
        limit=limit,
        workers=workers,
    )
    console.print(f"[green]Done[/green] wrote {len(written)} images to {output_dir}")
