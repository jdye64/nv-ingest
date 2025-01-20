# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


class PDFService:

    def __init__(self):
        self.pdf_parser = "pdfium"

    def ingest(self, input_directory):
        logger.info(f"Ingesting PDF Docs at : {input_directory}")

        # List all of the PDF files in the supplied directory
        pdf_files = [
            os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith(".pdf")
        ]

        logger.info(f"Ingesting PDF Files: {pdf_files}")

        for pdf_file in pdf_files:
            logger.info(f"Splitting PDF File: {pdf_file} into pages ...")

            self.split(pdf_file, os.path.dirname(pdf_file) + "/" + os.path.splitext(os.path.basename(pdf_file))[0])

            # pdf = pdfium.PdfDocument(pdf_file)
            # num_pages = len(pdf)
            # logger.info(f"Total Pages: {num_pages}")

            # for page_number in range(num_pages):
            #     # Get the page
            #     page = pdf[page_number]
            #     # Extract text from the page
            #     text = page.get_textpage().get_text_range()
            #     logger.info(f"Page {page_number + 1}:\n{text}\n")

    def split(self, input_pdf_path, output_dir):

        logger.info(f"Output Directory: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Directory '{output_dir}' created.")
        else:
            logger.info(f"Directory '{output_dir}' already exists.")

        pdf = pdfium.PdfDocument(input_pdf_path)
        num_pages = len(pdf)

        for page_index in range(num_pages):
            page_pdf = pdfium.PdfDocument.new()
            page_pdf.import_pages(pdf, pages=[page_index], index=0)

            output_path = f"{output_dir}/page_{page_index + 1}.pdf"
            page_pdf.save(output_path)
            logger.info(f"Saved: {output_path}")

            page_pdf.close()

        pdf.close()
