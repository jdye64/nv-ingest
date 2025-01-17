# Temporary testing script for common module ....
import logging
import sys

from src.nv_ingest_common.pdf.pdf_service import PDFService

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler(sys.stdout)],  # Write logs to stdout
)

pdf_input_dir = "/media/jeremy/storage/bo20"

pdf_service = PDFService()
ingestion_results = pdf_service.ingest(pdf_input_dir)
