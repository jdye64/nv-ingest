from typing import List
from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import write_to_nvingest_collection, create_nvingest_collection

import click

from utils import log, save_extracts, milvus_chunks, segment_results, embed_info


def ingest(
    hostname: str = "localhost",
    collection_name: str = "bo767_multimodal",
    input_files_glob: str = "/raid/rgelhausen/bo767/*.pdf",
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_images: bool = False,
    text_depth: str = "page",
    paddle_output_format: str = "markdown",
):

    model_name, dense_dim = embed_info()

    ingestion_start = time.time()
    ingestor = (
        Ingestor(message_client_hostname=hostname)
        .files(input_files_glob)
        .extract(
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_images=extract_images,
            text_depth=text_depth,
            paddle_output_format=paddle_output_format,
        )
        .embed()
        # .save_to_disk(output_directory="/tmp")
    )
    results, failures = ingestor.ingest(show_progress=True, return_failures=True, save_to_disk=True)
    print(failures)
    ingestion_end = time.time()
    ingestion_time = ingestion_end - ingestion_start
    log("ingestion_time", ingestion_time)
    log("ingestion_pages_per_sec", 54730 / ingestion_time)
    log("failure_count", len(failures))
    log("failures", failures)
    log("ingestion_files_per_sec", 767 / ingestion_time)
    log("record_count", len(results))

    save_extracts(results)

    indexing_start = time.time()
    sparse = False

    schema_start = time.time()
    gpu_search = True
    create_nvingest_collection("bo767_text", f"http://{hostname}:19530", sparse=sparse, gpu_search=gpu_search)
    create_nvingest_collection("bo767_tables", f"http://{hostname}:19530", sparse=sparse, gpu_search=gpu_search)
    create_nvingest_collection("bo767_charts", f"http://{hostname}:19530", sparse=sparse, gpu_search=gpu_search)
    schema_end = time.time()
    log("schema_creation", schema_end - schema_start)

    text_results, table_results, chart_results = segment_results(results)
    log("text_record_count", len(text_results))
    log("table_record_count", len(table_results))
    log("chart_record_count", len(chart_results))
    write_to_nvingest_collection(
        text_results,
        "bo767_text",
        sparse=sparse,
        milvus_uri=f"http://{hostname}:19530",
        minio_endpoint="localhost:9000",
    )
    write_to_nvingest_collection(
        table_results,
        "bo767_tables",
        sparse=sparse,
        milvus_uri=f"http://{hostname}:19530",
        minio_endpoint="localhost:9000",
    )
    write_to_nvingest_collection(
        chart_results,
        "bo767_charts",
        sparse=sparse,
        milvus_uri=f"http://{hostname}:19530",
        minio_endpoint="localhost:9000",
    )

    multimodal_index_start = time.time()
    create_nvingest_collection(collection_name, f"http://{hostname}:19530", sparse=sparse, gpu_search=gpu_search)
    write_to_nvingest_collection(
        results,
        collection_name,
        sparse=sparse,
        milvus_uri=f"http://{hostname}:19530",
        minio_endpoint="localhost:9000",
        model_name=model_name,
        dense_dim=dense_dim,
    )
    multimodal_index_end = time.time()
    multimodal_indexing_time = multimodal_index_end - multimodal_index_start
    log("multimodal_indexing_time", multimodal_indexing_time)
    log("total_indexing_time", time.time() - indexing_start)
    log("e2e_runtime", ingestion_time + multimodal_indexing_time)
    log("e2e_pages_per_sec", 54730 / (ingestion_time + multimodal_indexing_time))
    for col in ["text", "tables", "charts", "multimodal"]:
        milvus_chunks(f"bo767_{col}")


@click.command()
@click.option("--client_host", default="localhost", help="DNS name or URL for the endpoint.")
@click.option("--client_port", default=7670, type=int, help="Port for the client endpoint.")
@click.option(
    "--concurrency_n", default=10, show_default=True, type=int, help="Number of inflight jobs to maintain at one time."
)
@click.option("--dry_run", is_flag=True, help="Perform a dry run without executing actions.")
@click.option("--fail_on_error", is_flag=True, help="Fail on error.")
@click.option("--output_directory", type=click.Path(), default=None, help="Output directory for results.")
# @click.option(
#     "--save_images_separately",
#     is_flag=True,
#     help="Save images separately from returned metadata. This can make metadata files more human readable",
# )
# @click.option(
#     "--collect_profiling_traces",
#     is_flag=True,
#     default=False,
#     help="""
# \b
# If enabled the CLI will collect the 'profile' for each file that was submitted to the
# nv-ingest REST endpoint for processing.


# \b
# Those `trace_id` values will be consolidated and then a subsequent request will be made to
# Zipkin to collect the traces for each individual `trace_id`. The trace is rich with information
# that can further breakdown the runtimes for each section of the codebase. This is useful
# for locating portions of the system that might be bottlenecks for the overall runtimes.
# """,
# )
@click.pass_context
def main(
    ctx,
    client_host: str,
    client_port: int,
    concurrency_n: int,
    dry_run: bool,
    fail_on_error: bool,
    output_directory: str,
):
    try:
        print(f"Starting this bad boy ...")

    except Exception as err:
        logging.error(f"Error: {err}")
        raise


if __name__ == "__main__":
    main()
