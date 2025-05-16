import os
import sys
import ray
import logging
from typing import List
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import SpanContext, TraceFlags, SpanKind

# Initialize Ray
ray.init()


def get_tracer():
    """Get a tracer instance for the current process."""
    resource = Resource(attributes={"service.name": "nv-ingest"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317", insecure=True))
    trace.get_tracer_provider().add_span_processor(span_processor)
    return tracer


@ray.remote
def process_pdf(pdf_path: str, context_carrier: dict) -> None:
    """
    Process a single PDF file using PDFium.

    Args:
        pdf_path: Path to the PDF file
        context_carrier: OpenTelemetry context carrier for tracing
    """
    # Get a new tracer instance for this process
    tracer = get_tracer()
    propagator = TraceContextTextMapPropagator()

    # Extract context from carrier
    ctx = propagator.extract(carrier=context_carrier)
    token = attach(ctx)

    try:
        # Create a span that's linked to the parent span from the context
        with tracer.start_as_current_span(
            "process_pdf", context=ctx, kind=SpanKind.CLIENT, attributes={"pdf_path": pdf_path}
        ) as span:
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

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        raise
    finally:
        # always detach the context
        detach(token)


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

    # Get a tracer for the main process
    tracer = get_tracer()
    propagator = TraceContextTextMapPropagator()

    # Create the root span
    with tracer.start_as_current_span(
        "pdf_job", kind=SpanKind.SERVER, attributes={"directory_path": directory_path}
    ) as root_span:
        # Get list of PDF files
        pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

        root_span.set_attribute("pdf_file_count", len(pdf_files))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return

        # Create context carrier for child spans
        context_carrier = {}
        propagator.inject(carrier=context_carrier)

        # Launch Ray tasks for each PDF
        tasks = [process_pdf.remote(pdf_file, context_carrier) for pdf_file in pdf_files]

        # Wait for all tasks to complete
        ray.get(tasks)

        logger.info(f"Completed processing {len(pdf_files)} PDF files")


if __name__ == "__main__":
    try:
        directory = "/home/jdyer/datasets/bo20"
        process_pdf_directory(directory)

        # Flush all traces
        trace.get_tracer_provider().shutdown()
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        sys.exit(1)
