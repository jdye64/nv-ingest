# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import time
from io import BytesIO
from nv_ingest_api.util.pdf.pdfium import extract_simple_images_from_pdfium_page
import pypdfium2 as libpdfium

from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.pass_context
def main(
    ctx,
):
    print(f"Running kiss.")
    start_time = time.time()
    dataset_path = Path("/media/jeremy/storage/bo20")

    for pdf in dataset_path.iterdir():
        if pdf.is_file() and pdf.suffix == ".pdf":
            doc = libpdfium.PdfDocument(pdf)
            print(pdf)

            pdf_pages = len(doc)
            for page_idx in range(pdf_pages):
                page = doc.get_page(page_idx)
                page_width, page_height = page.get_size()
                print(f"Page {page_idx} size: {page_width} x {page_height}")

                # The text from the page
                textpage = page.get_textpage()

                # Get the images from the page
                images = extract_simple_images_from_pdfium_page(page, max_depth=2)
                print(f"Images: {len(images)}")
            doc.close()

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
