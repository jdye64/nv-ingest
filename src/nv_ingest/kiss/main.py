import os
import sys
import ray
import logging
from typing import List
from opentelemetry import trace, context, propagate
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

# Initialize Ray
ray.init()


def get_tracer():
    """Get a tracer instance for the current process."""
    resource = Resource(attributes={"service.name": "nv-ingest"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Use OTLP exporter for sending traces to collector
    exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    span_processor = BatchSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return tracer


@ray.remote
def process_pdf_remote(pdf_path: str, context_carrier: dict) -> None:
    """
    Process a single PDF file using PDFium in a remote task.

    Args:
        pdf_path: Path to the PDF file
        context_carrier: OpenTelemetry context carrier for tracing
    """
    # Get a new tracer instance for this process
    tracer = get_tracer()

    # Extract context from carrier
    ctx = propagate.extract(carrier=context_carrier)
    token = context.attach(ctx)

    try:
        # Create a span that's linked to the parent span from the context
        with tracer.start_as_current_span(
            "pdfium_processing", context=ctx, kind=SpanKind.CLIENT, attributes={"pdf_path": pdf_path}
        ) as span:
            # Add some work to make the span more visible
            span.set_attribute("processing_started", True)
            # Add an event to make the span more visible
            span.add_event("Processing started")

            # Load and process PDF with PDFium
            pdf = pdfium.PdfDocument(pdf_path)

            # Get basic PDF info
            n_pages = len(pdf)
            span.set_attribute("page_count", n_pages)

            logger.info(f"Processing PDF {pdf_path} with {n_pages} pages")

            # Process each page
            for page_idx in range(n_pages):
                page = pdf.get_page(page_idx)
                # Add page processing logic here
    finally:
        context.detach(token)
        # Force flush the span processor
        trace.get_tracer_provider().force_flush()


def process_pdf_directory(directory_path: str) -> None:
    """
    Process all PDFs in the given directory using Ray.

    Args:
        directory_path: Path to directory containing PDF files
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    tracer = get_tracer()

    # Create the root span
    with tracer.start_as_current_span(
        "pdf_processing", kind=SpanKind.SERVER, attributes={"directory_path": directory_path}
    ) as root_span:
        # Create context carrier for propagation
        context_carrier = {}
        propagate.inject(carrier=context_carrier)

        print(f"Context carrier: {context_carrier}")
        print(f"Root span context: {root_span.get_span_context()}")

        # Get list of PDF files
        pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

        root_span.set_attribute("pdf_file_count", len(pdf_files))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return

        # Launch Ray tasks for each PDF
        tasks = [process_pdf_remote.remote(pdf_file, context_carrier) for pdf_file in pdf_files]

        # Wait for all tasks to complete
        ray.get(tasks)

        logger.info(f"Completed processing {len(pdf_files)} PDF files")

        # Force flush the span processor
        trace.get_tracer_provider().force_flush()


if __name__ == "__main__":
    try:
        directory = "/home/jdyer/datasets/bo20"
        process_pdf_directory(directory)

        # Flush all traces
        trace.get_tracer_provider().shutdown()
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        sys.exit(1)
