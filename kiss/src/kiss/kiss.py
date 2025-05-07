# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import pickle
import logging
import time
from io import BytesIO
from nv_ingest_api.util.pdf.pdfium import extract_simple_images_from_pdfium_page
import pypdfium2 as libpdfium
import os

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
    base_extracts_path = Path("/media/jeremy/storage/bo20_extracts")

    for pdf in dataset_path.iterdir():
        if pdf.is_file() and pdf.suffix == ".pdf":
            pdf_basename = os.path.splitext(os.path.basename(pdf))[0]
            doc = libpdfium.PdfDocument(pdf)
            print(pdf_basename)

            pdf_pages = len(doc)
            for page_idx in range(pdf_pages):
                page = doc.get_page(page_idx)
                page_width, page_height = page.get_size()
                print(f"Page {page_idx} size: {page_width} x {page_height}")

                # # The text from the page
                # textpage = page.get_textpage()

                # Get the images from the page
                images = extract_simple_images_from_pdfium_page(page, max_depth=2)
                print(f"Images: {len(images)}")

                # Save all of the base64 encoded images to a local directory
                pdf_path = Path(str(base_extracts_path) + "/" + pdf_basename + "/page_" + str(page_idx))
                os.makedirs(pdf_path, exist_ok=True)

                for image_idx in range(len(images)):
                    filename = str(pdf_path) + "/" + str(image_idx) + ".pkl"
                    with open(filename, "wb") as f:
                        pickle.dump(images[image_idx], f)

            doc.close()

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
