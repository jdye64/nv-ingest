# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import click
import json
import os
import pandas as pd

import logging, time

from util.io import log_zipkin_json_file
import plotly.express as px

from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.util.file_processing.extract import extract_file_content
from zipkin import AsyncZipkinClient

logger = logging.getLogger(__name__)


# def ingest_documents(
#         _client: NvIngestClient,
#         input_directory: str,
#      ) -> List[Tuple[str, str]]:
#     """
#     Indexes all of the PDFs in the specified directory against the Retriever
#     service for the collection specified.

#     Parameters:
#         _client: Preconfigured client ready to
#             communicate with the service.
#         input_directory (str): Local directory where PDFs will be
#             listed from, files only.

#     Returns:
#         List of trace_ids for each file that was indexed
#     """
#     try:
#         total_files = 0

#         try:
#             pdf_files = glob.glob(input_directory + "/*.pdf")
#             total_files = sum(
#                 1 for pdf_file in pdf_files
#                 if os.path.isfile(os.path.join(input_directory, pdf_file))
#             )
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             return (None, None)

#         job_and_trace_ids = []
#         start_time = time.time()

#         print(f"Begin indexing {total_files} PDF(s) to Nv-Ingest REST endpoint: {start_time}")
        
#         # Loop through and submit each PDF file
#         for pdf in pdf_files:
#             print(f"PDF File: {pdf}")
#             file_content, file_type = extract_file_content(pdf)

#             # A JobSpec is an object that defines a document and how it should
#             # be processed by the nv-ingest service.
#             job_spec = JobSpec(
#                 document_type=file_type,
#                 payload=file_content,
#                 source_id=pdf,
#                 source_name=pdf,
#                 extended_options=
#                 {
#                     "tracing_options":
#                     {
#                         "trace": True,
#                         "ts_send": time.time_ns()
#                     }
#                 }
#             )

#             # configure desired extraction modes here. Multiple extraction
#             # methods can be defined for a single JobSpec
#             extract_task = ExtractTask(
#                 document_type=file_type,
#                 extract_text=True,
#                 extract_images=True,
#                 extract_tables=True
#             )

#             job_spec.add_task(extract_task)
#             job_id = _client.add_job(job_spec)
#             trace_ids = _client.submit_job(job_id, "morpheus_task_queue")
#             job_and_trace_ids.append((job_id, trace_ids[0]))
#             # print(f"Trace-Ids: {trace_ids}")
#             # result = client.fetch_job_result(job_id, timeout=60)
#             # print(f"Got {len(result)} results")

#         # trace_ids = asyncio.run(_client.index_files_to_collection(
#         #     file_paths=pdf_files,
#         #     collection_id=collection_id
#         # ))

#         # failed_docs = trace_ids.count(None)

#         print(f"Indexing {len(trace_ids)} PDF(s) took {time.time() - start_time}.")
#         return pdf_files, job_and_trace_ids
        
#     except Exception as err:
#         logger.error(f"Error: {err}")
#         raise
    
def fetch_all_results(client, job_ids):
    completed_job_ids = []
    for job_id in job_ids:
        try:
            print(f"Checking Job with ID: {job_id}")
            client.fetch_job_result(job_id, timeout=60)
            completed_job_ids.append(job_id)
            print(f"Received results for job_id: {job_id} - {len(job_ids) - len(completed_job_ids)} remaining")
        except Exception:
            print(f"Timeout for job_id: {job_id} ... keep trying")
            
        
def aggregate_results(zipkin_file):
    print(f"Aggregating Results from Zipkin file: {zipkin_file}")
    bench = json.loads(open("zipkin.json").read())
    fns = list(bench.keys())
    breakdowns = []
    for fn in fns:
        print(f"PDF File: {fn}")
        [breakdowns.append({"name": x["name"], "duration": x["duration"]/1_000_000}) for x in bench[fn]]
        print(breakdowns[0])
    df = pd.DataFrame(breakdowns)
    # print(df.dtypes)
    # print(df.columns)
    # breakpoint()
    # fig = px.bar(df, x="name", y="duration")
    # fig.show()
    cached_sum = df[df['name'].str.startswith('cached_')]['duration'].sum()
    yolox_sum = df[df['name'].str.startswith('yolox_')]['duration'].sum()
    deplot_sum = df[df['name'].str.startswith('google/deplot_')]['duration'].sum()
    paddle_sum = df[df['name'].str.startswith('paddle_')]['duration'].sum()
    
    agg_breakdowns = []
    agg_breakdowns.append({"name": "Cached", "duration": cached_sum})
    agg_breakdowns.append({"name": "Yolox", "duration": yolox_sum})
    agg_breakdowns.append({"name": "Deplot", "duration": deplot_sum})
    agg_breakdowns.append({"name": "PaddleOCR", "duration": paddle_sum})
    agg_df = pd.DataFrame(agg_breakdowns)
    
    fig = px.bar(agg_df, x="name", y="duration")
    fig.show()
    
    # # Keys of the dict are the path (including name) of each submitted PDF
    # with open(zipkin_file, 'r') as f:
    #     data = json.load(f)
        
    # for pdf_file in data.keys():
    #     # List of all the traces for this PDF file
    #     pdf_file_data = data[pdf_file]
    #     df = pd.DataFrame(pdf_file_data)
    #     print(df.columns)
    #     breakpoint()
        
    #     # Aggregate total duration for each name
    #     total_duration = df.groupby('name')['duration'].sum().reset_index()

    #     # Rename the 'duration' column to 'total_duration' for clarity
    #     total_duration.rename(columns={'duration': 'total_duration_ns'}, inplace=True)
        
    #     # Convert total_duration to milliseconds and create a new column
    #     total_duration['total_duration_ms'] = total_duration['total_duration'] / 1_000_000  # Convert nanoseconds to milliseconds
        
    #     # Create another column for total runtime in seconds
    #     total_duration['total_runtime_s'] = total_duration['total_duration'] / 1_000_000_000  # Convert nanoseconds to seconds
        
    #     # Limit decimal places to 4
    #     total_duration['total_duration_ms'] = total_duration['total_duration_ms'].round(4)
    #     total_duration['total_runtime_s'] = total_duration['total_runtime_s'].round(4)
        
    #     # Sort the DataFrame by 'total_duration' in descending order
    #     total_duration_sorted = total_duration.sort_values(by='total_duration', ascending=False)

    #     # Display the aggregated DataFrame
    #     print(total_duration_sorted)

@click.command()
@click.option(
    "--input_directory",
    # default="/media/jeremy/storage/ingest_smoke_test",
    default="/media/jeremy/storage/image_pipeline_test_20",
    # default="/media/jeremy/storage/bo767",
    help="Directory containing PDFs to send to NeMo Retriever"
)
@click.option(
    "--client_host",
    # default="https://ingest.dev.aire.nvidia.com",
    # default="http://ipp1-3304.ipp1u1.colossus.nvidia.com",
    default="http://10.78.7.115",
    help="DNS name or URL for the endpoint."
)
@click.option(
    "--client_port",
    default=7670,
    help="Port where nv-ingest service is running"
)
@click.option(
    "--zipkin_port",
    default=9411,
    help="Port where zipkin service is running"
)
@click.option(
    "--output_directory",
    type=click.Path(),
    default=os.getcwd(),
    help="Output directory for results."
)
@click.option(
    "--concurrent_requests",
    default=1,
    help="The number of concurrent PDF documents that should be submitted to the indexing endpoint"
)
@click.pass_context
def main(
    ctx,
    input_directory: str,
    client_host: str,
    client_port: int,
    zipkin_port: int,
    output_directory: str,
    concurrent_requests: int,
):

    print("===== Nv-Ingest Profiling Tool =====")
    
    client = NvIngestClient(
        message_client_hostname=client_host, # Host where nv-ingest-ms-runtime is running
        message_client_port=client_port
    )
    
    pdf_files, job_and_trace_ids = ingest_documents(
        _client=client,
        input_directory=input_directory
    )
    
    job_ids = [x[0] for x in job_and_trace_ids]
    trace_ids = [x[1] for x in job_and_trace_ids]
    
    fetch_all_results(client, job_ids)
    
    sleep_seconds = 15
    print(f"Sleeping {sleep_seconds} seconds before getting Zipkin traces")
    time.sleep(sleep_seconds)
    zipkin_client = AsyncZipkinClient(client_host, zipkin_port, concurrent_requests)
    traces = asyncio.run(zipkin_client.get_metrics(trace_ids=trace_ids))
    print(f"Traces: {traces}")
    
    log_zipkin_json_file(traces, pdf_files, output_directory=output_directory)
    
    # Aggregate the results
    aggregate_results(output_directory + "/zipkin.json")


if __name__ == "__main__":
    main()
