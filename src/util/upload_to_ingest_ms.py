# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import base64
import glob
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from statistics import mean
from statistics import median

import chardet
import click
import fitz
from tqdm import tqdm

from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.ingest_job_schema import validate_ingest_job
from nv_ingest.util.redis import RedisClient

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY", None)
UNSTRUCTURED_URL = os.environ.get("UNSTRUCTURED_URL", "http://localhost:8003")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", None)
EXTRACTABLE_FILE_TYPES = ["pdf"]


def configure_logging(level_name):
    """
    Configures the global logging level based on a string name.

    Parameters:
    - level_name (str): The name of the logging level (e.g., "DEBUG", "INFO").
    """
    global logger

    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(numeric_level)


def setup_global_executor(n_workers, concurrency_mode="thread", use_dask=False):
    """
    Initializes the global ThreadPoolExecutor with the specified number of worker threads.

    Parameters:
    - n_workers (int): The number of worker threads to use in the ThreadPoolExecutor.
    """

    if use_dask:
        from dask.distributed import Client

        if n_workers == 0:
            n_workers = None  # Use default
        if concurrency_mode == "thread":
            executor = Client(
                n_workers=n_workers, threads_per_worker=1, processes=False
            )
        else:
            executor = Client(n_workers=n_workers, threads_per_worker=1)
        logger.debug(f"Global dask client initialized with {n_workers} workers.")
    else:
        executor_class = (
            ThreadPoolExecutor if concurrency_mode == "thread" else ProcessPoolExecutor
        )
        executor = executor_class(max_workers=n_workers)
        logger.debug(
            f"Global {str(executor_class)} initialized with {n_workers} workers."
        )

    return executor


def estimate_page_count(file_path):
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == ".pdf":
            # Handle PDF files with PyMuPDF
            with fitz.open(file_path) as doc:
                return len(doc)
        elif file_extension.lower() in [".txt", ".md"]:
            # Handle text files
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                word_count = len(text.split())
                pages_estimated = word_count / 300
                return round(pages_estimated)
        else:
            print(f"Unsupported file type: {file_extension}")
            return 1

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None


def build_extraction_tasks(methods, file_type):
    """
    Constructs a list of extraction tasks based on specified methods and the file type.

    Dynamically creates tasks for extracting content from files, tailored to specific
    methods and the type of file being processed. Supports integration with various
    extraction technologies by configuring method-specific properties.

    Parameters
    ----------
    methods : list of str
        The extraction methods to be applied. Possible values include 'pymupdf',
        'haystack', 'unstructured_local', 'unstructured_service', and 'llama_parse'.
    file_type : str
        The type of the file to be processed (e.g., 'pdf', 'txt').

    Returns
    -------
    list of dict
        A list of dictionaries, each representing an extraction task to be executed.
        Each dictionary includes the task type, the method to be used, the document type,
        and other parameters specific to the extraction method.

    Notes
    -----
    - The function adapts to different extraction methods by applying common and
      method-specific properties.
    - It returns an empty list if no valid methods are specified or if the file type
      does not require extraction.
    """

    tasks = []
    if (not methods) or (file_type not in EXTRACTABLE_FILE_TYPES):
        logger.debug(
            "No extraction tasks specified or file type does not require extraction."
        )
        return tasks

    common_properties = {
        "extract_text": True,
        "extract_images": True,
        "extract_tables": False,
        "text_depth": "document",
    }

    # Define default properties for unstructured tasks
    unstructured_properties = {
        "api_key": UNSTRUCTURED_API_KEY,
        "unstructured_url": UNSTRUCTURED_URL,
    }

    # Define default properties for LlamaParse tasks
    llama_parse_properties = {
        "api_key": LLAMA_CLOUD_API_KEY,
    }

    # Add other task types based on _extract_methods
    for method in methods:
        task_props = common_properties.copy()
        task = {
            "type": "extract",
            "task_properties": {
                "method": method,
                "document_type": file_type,
                "params": task_props,
            },
        }

        if method in ["unstructured-local", "unstructured-service"]:
            task["task_properties"]["params"].update(unstructured_properties)
        elif method in ["llama_parse"]:
            task["task_properties"]["params"].update(llama_parse_properties)
        else:
            pass  # Others

        # logger.debug('Adding task: %s', json.dumps(task, indent=2))
        tasks.append(task)

    return tasks


def extract_file_content(path):
    """
    Extracts content from a file, supporting different formats like PDF and text files.

    For PDF files, reads the content as binary and encodes it in base64. For text files,
    detects the encoding and reads the content accordingly. This approach ensures that
    the extracted content is appropriately handled according to the file's format.

    Parameters
    ----------
    path : str
        The path to the file from which content is to be extracted.

    Returns
    -------
    str
        The extracted content of the file. For PDFs, the content is base64 encoded;
        for text files, it is returned in plain text, using the detected encoding.

    Raises
    ------
    ValueError
        If the file type is unsupported, indicating that the function cannot process
        this type of file.

    Notes
    -----
    - This function is designed to handle PDF and text files but can be extended to support
      additional file types.
    - It uses a simple heuristic for encoding detection in text files to maximize compatibility.
    """

    document_type = os.path.basename(path).split(".")[-1].lower()
    if (
        document_type not in DocumentTypeEnum.__members__
    ):  # Check if file type is supported
        raise ValueError(f"Unsupported file type: {document_type}")

    if document_type != DocumentTypeEnum.txt.name.lower():
        # For PDF files, read as binary and encode in base64
        with open(path, "rb") as file:
            encoding = "utf-8"
            content = base64.b64encode(file.read()).decode(encoding)
            logger.debug(
                f"Encoded {document_type} content: {content[:100]}... (truncated)"
            )
    else:
        # Detect encoding for non-PDF files
        with open(path, "rb") as file:
            raw_data = file.read(50000)
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        # Re-open the file with the detected encoding
        with open(path, "r", encoding=encoding) as file:
            content = file.read()
            logger.debug(
                f"Read plain text content with detected encoding ({encoding}): "
                f"{content[:100]}... (truncated)"
            )

    logger.debug(f"Content length: {len(content)}")
    return content, DocumentTypeEnum[document_type]


def process_file(file_path):
    """
    Synchronously processes a single file, extracting its content and collecting file details.

    This function serves as a high-level interface for file processing, invoking content
    extraction and aggregating the results along with file metadata. It is designed to work
    within a larger processing pipeline, providing necessary data for subsequent tasks or
    storage.

    Parameters
    ----------
    file_path : str
        The path to the file that needs to be processed.

    Returns
    -------
    dict
        A dictionary containing details about the processed file, including its name, a unique
        identifier, the extracted content, and the document type.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during the file processing, signaling issues with
        content extraction or file handling.

    Notes
    -----
    - The function directly utilizes `extract_file_content` for content extraction and performs
      basic error handling.
    - It constructs a simple metadata object that can be utilized for further processing or
      logging.
    """

    try:
        file_name = os.path.basename(file_path)
        content, document_type = extract_file_content(
            file_path
        )  # Call the synchronous function directly

        return {
            "source_name": file_name,
            "source_id": file_name,
            "content": content,
            "document_type": document_type,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def load_data_from_path(path):
    """
    Loads data from a specified file path, preparing it for processing.

    Parameters
    ----------
    path : str
        The path to the file from which data should be loaded.

    Returns
    -------
    dict
        A dictionary containing keys 'file_name', 'id', 'content', and 'document_type',
        each of which maps to a list that includes the respective details for the processed file.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the specified path is not a file.

    Notes
    -----
    This function is designed to load and prepare file data for further processing,
    packaging the loaded data along with metadata such as file name and document type.
    """

    result = {"source_name": [], "source_id": [], "content": [], "document_type": []}

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    if not os.path.isfile(path):
        raise ValueError("The provided path is not a file.")

    file_data = process_file(file_path=path)
    result["content"].append(file_data["content"])
    result["document_type"].append(file_data["document_type"])
    result["source_name"].append(file_data["source_name"])
    result["source_id"].append(file_data["source_id"])

    return result


def debug_print_job_payload(job_payload):
    """
    Prints a summarized version of the job payload for debugging purposes.

    Parameters
    ----------
    job_payload : dict
        The job payload to be printed, typically containing detailed task or job information.

    Notes
    -----
    The function deep-copies the input payload to avoid modifying the original data,
    then shortens the 'content' field for brevity before printing.
    """

    import copy

    payload = copy.deepcopy(job_payload)
    payload["job_payload"]["content"][0] = (
        payload["job_payload"]["content"][0][:20] + "..."
    )
    logger.info(json.dumps(payload, indent=2))


def submit_job_and_wait_for_response(
    redis_client, job_data, tasks, task_queue, timeout=90
):
    """
    Submits a job consisting of multiple tasks to the Redis task queue and waits for a response.

    Parameters
    ----------
    redis_client : redis.Redis
        The Redis client instance used to communicate with the Redis server.
    job_data : dict
        The primary data for the job to be submitted.
    tasks : list
        A list of tasks that constitute the job.
    task_queue : str
        The name of the Redis task queue to which the job is submitted.
    timeout : int, optional
        The maximum time to wait for a response, in seconds. Defaults to 90.

    Returns
    -------
    dict
        The response data received from processing the job, parsed as a dictionary.

    Raises
    ------
    RuntimeError
        If no response is received within the specified timeout period, indicating a potential
        issue with job processing or response delivery.

    Notes
    -----
    This function constructs a job descriptor, submits it to the specified Redis task queue,
    and waits for a response within a configurable timeout. It raises an exception if no response
    is received.
    """

    job_id = str(uuid.uuid4())
    job_desc = {
        "job_payload": job_data,
        "job_id": job_id,
        "tasks": tasks,
        "tracing_options": {"trace": True, "ts_send": time.time_ns()},
    }

    # debug_print_job_payload(job_desc)
    validate_ingest_job(job_desc)

    response_channel = f"response_{job_id}"
    response_channel_expiration = int(
        timeout * 1.05
    )  # Set expiration to timeout+grace period

    # Use submit_job method of RedisClient
    try:
        response = redis_client.submit_job(
            task_queue,
            job_desc,
            response_channel,
            response_channel_expiration,
            timeout=timeout,
        )
    except Exception:
        traceback.print_exc()
        raise

    if response:
        return response
    else:
        raise RuntimeError("No response received within timeout period")


def generate_matching_files(file_sources):
    """
    Generates a list of file paths that match the given patterns specified in file_sources.

    Parameters
    ----------
    file_sources : list of str
        A list containing the file source patterns to match against.

    Returns
    -------
    generator
        A generator yielding paths to files that match the specified patterns.

    Notes
    -----
    This function utilizes glob pattern matching to find files that match the specified patterns.
    It yields each matching file path, allowing for efficient processing of potentially large
    sets of files.
    """

    files = [
        file_path
        for pattern in file_sources
        for file_path in glob.glob(pattern, recursive=True)
        if os.path.isfile(file_path)
    ]
    for file_path in files:
        yield file_path


def process_source(
    source,
    id,
    redis_host,
    redis_port,
    task_queue,
    extract,
    extract_method,
    split,
    split_params,
):
    """
    Processes a single source file according to the specified parameters, submitting the
    processing tasks to Redis.

    Parameters
    ----------
    source : str
        The path to the source file to be processed.
    id : Any
        An identifier associated with the processing of this source file.
    redis_host : str
        The hostname of the Redis server.
    redis_port : int
        The port number of the Redis server.
    task_queue : str
        The Redis task queue where processing tasks should be submitted.
    extract : bool
        Indicates whether content extraction should be performed.
    extract_method : list of str
        Specifies the methods to be used for content extraction.
    split : bool
        Indicates whether the content should be split according to specific properties.

    Raises
    ------
    RuntimeError
        Signifies that this function should be implemented or is a placeholder.

    Notes
    -----
    This function is intended as a higher-level abstraction for processing source files,
    relying on lower-level processing functions. It should be implemented to fit the specific
    needs of the application.
    """
    redis_client = RedisClient(host=redis_host, port=redis_port)
    return _process_source(
        source,
        id,
        redis_client,
        task_queue,
        extract,
        extract_method,
        split,
        split_params,
    )


def _process_source(
    source, redis_client, task_queue, extract, extract_method, split, split_params
):
    """
    Processes a single source file by applying specified tasks such as splitting and extracting
    content, and submits these tasks along with the job data to a specified Redis task queue.

    Parameters
    ----------
    source : str
        The path to the source file to be processed.
    redis_client : Redis
        An instance of the Redis client connected to the Redis server.
    task_queue : str
        The name of the Redis task queue where the job will be submitted.
    extract : bool
        Flag indicating whether content extraction should be performed.
    extract_method : list of str
        A list of methods to be used for content extraction, applicable if `extract` is True.
    split : bool
        Flag indicating whether the content should be split according to specified properties.

    Returns
    -------
    tuple
        A tuple containing the response from the task submission and the size of the data
        processed from the source file in bytes.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during task preparation or job submission.
    """

    tasks = []
    if split:
        tasks.append({"type": "split", "task_properties": split_params})

    if extract:
        file_type = os.path.basename(source).split(".")[-1].lower()
        extract_tasks = build_extraction_tasks(extract_method, file_type)
        tasks.extend(extract_tasks)

    data_size = os.path.getsize(source)

    try:
        job_data = load_data_from_path(source)
        response = submit_job_and_wait_for_response(
            redis_client, job_data, tasks, task_queue, timeout=300
        )

        return response, data_size
    except Exception:
        return None, 0


def match_and_validate_files(file_source):
    """
    Matches and validates files based on the provided file source patterns.

    Parameters
    ----------
    file_source : list of str
        A list containing file source patterns to match against.

    Returns
    -------
    list of str or None
        A list of matching file paths if any matches are found; otherwise, None.
    """

    matching_files = list(generate_matching_files(file_source))
    if not matching_files:
        logger.warning("No files found matching the specified patterns.")
        return None
    return matching_files


def submit_tasks(executor, matching_files, processing_function, processing_args):
    """
    Submits processing tasks for each matching file to an executor.

    Parameters
    ----------
    executor : Executor
        The executor to which tasks will be submitted.
    matching_files : list of str
        A list of file paths to be processed.
    processing_function : callable
        The processing function to be applied to each file.
    processing_args : tuple
        Arguments to be passed to the processing function.

    Returns
    -------
    dict
        A dictionary mapping futures to source file paths.
    """

    return {
        executor.submit(processing_function, source, *processing_args): {
            "source": source,
            "pages": estimate_page_count(
                source
            ),  # Short term hack to get pages/sec -- needs to go into service
        }
        for source in matching_files
    }


def handle_future_result(future, future_data, output_directory, stage_elapsed_times):
    source = future_data["source"]
    try:
        response, data_processed = future.result()
        if response is None:
            logger.error(f"Error processing file {source}: No response received.")
            return None, 0  # Indicates an error; no data processed

        if output_directory:
            save_response_data(response, source, output_directory)

        process_response(response, stage_elapsed_times)
        return response, data_processed
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing file {source}: {str(e)}")
        return None, 0  # Indicates an error; no data processed


def save_response_data(response, source, output_directory):
    response_data = json.loads(response["data"])
    doc_meta_base = response_data[0]["metadata"]
    source_meta = doc_meta_base["source_metadata"]
    doc_name = source_meta["source_id"]
    output_name = f"{doc_name}.metadata.json"

    doc_map = organize_documents_by_type(response_data)
    for doc_type, documents in doc_map.items():
        doc_type_path = os.path.join(output_directory, doc_type)
        if not os.path.exists(doc_type_path):
            os.makedirs(doc_type_path)

        with open(os.path.join(doc_type_path, output_name), "w") as f:
            f.write(json.dumps(documents, indent=2))


def organize_documents_by_type(response_data):
    doc_map = {}
    for document in response_data:
        doc_meta = document["metadata"]
        doc_content_metadata = doc_meta["content_metadata"]
        doc_type = doc_content_metadata["type"]
        if doc_type not in doc_map:
            doc_map[doc_type] = []
        doc_map[doc_type].append(document)
    return doc_map


def process_tasks(future_to_file, output_directory, progress_bar, start_time_ns):
    stage_elapsed_times = defaultdict(list)
    total_data_processed = 0
    total_pages_processed = 0

    for future in as_completed(future_to_file):
        progress_bar.update(1)
        future_data = future_to_file.pop(future)  # Cleanup as tasks complete
        pages = future_data["pages"]
        total_pages_processed += pages

        response, data_processed = handle_future_result(
            future, future_data, output_directory, stage_elapsed_times
        )
        if response is None:
            continue  # Skip to next future if there was an error

        total_data_processed += data_processed

        # Calculate and update performance metrics
        elapsed_time_ns = time.time_ns() - start_time_ns
        elapsed_time = elapsed_time_ns / 1e9
        pages_per_sec = total_pages_processed / elapsed_time if elapsed_time > 0 else 0
        progress_bar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}", refresh=True)

    return stage_elapsed_times, total_data_processed, total_pages_processed


def process_response(response, stage_elapsed_times):
    """
    Process the response to extract trace data and calculate elapsed time for each stage.

    Parameters
    ----------
    response : dict
        The response dictionary containing trace information for processing stages.
    stage_elapsed_times : defaultdict(list)
        A defaultdict to accumulate elapsed times for each processing stage.

    Notes
    -----
    The function iterates over trace data in the response, identifying entry and exit times for
    each stage, and calculates the elapsed time which is then appended to the respective stage in
    `stage_elapsed_times`.
    """

    trace_data = response.get("trace", {})
    for key, entry_time in trace_data.items():
        if "entry" in key:
            exit_key = key.replace("entry", "exit")
            exit_time = trace_data.get(exit_key)
            if exit_time:
                stage_name = key.split("::")[2]
                elapsed_time = exit_time - entry_time
                stage_elapsed_times[stage_name].append(elapsed_time)


def report_stage_statistics(stage_elapsed_times, total_trace_elapsed, abs_elapsed):
    """
    Reports the statistics for each processing stage, including average, median, and total time
    spent.

    Parameters
    ----------
    stage_elapsed_times : defaultdict(list)
        A defaultdict containing lists of elapsed times for each processing stage.
    total_trace_elapsed : float
        The total elapsed time across all processing stages.
    abs_elapsed : float
        The absolute elapsed time from the start to the end of processing.

    Notes
    -----
    This function logs the average, median, total time, and the percentage of total computation
    for each processing stage. It also calculates and logs the unresolved time, if any.
    """

    for stage, times in stage_elapsed_times.items():
        avg_time = mean(times)
        med_time = median(times)
        total_stage_time = sum(times)
        percent_of_total = (
            (total_stage_time / total_trace_elapsed) * 100
            if total_trace_elapsed > 0
            else 0
        )
        logger.info(
            f"{stage}: Avg: {avg_time / 1e6:.2f} ms, Median: {med_time / 1e6:.2f} ms, "
            f"Total Time: {total_stage_time / 1e6:.2f} ms, "
            f"Total % of Trace Computation: {percent_of_total:.2f}%"
        )

    unresolved_time = abs_elapsed - total_trace_elapsed
    if unresolved_time > 0:
        percent_unresolved = (unresolved_time / abs_elapsed) * 100
        logger.info(
            f"Unresolved time: {unresolved_time / 1e6:.2f} ms, "
            f"Percent of Total Elapsed: {percent_unresolved:.2f}%"
        )
    elif unresolved_time <= 0:
        logger.info(
            "No unresolved time detected. Trace times account for the entire elapsed duration."
        )


def report_overall_speed(
    total_data_processed, total_pages_processed, start_time_ns, total_files
):
    """
    Reports the overall processing speed and the total data processed.

    Parameters
    ----------
    total_data_processed : int
        The total amount of data processed, in bytes.
    start_time_ns : int
        The start time of the processing, in nanoseconds.
    total_files : int
        The total number of files processed.

    Notes
    -----
    This function calculates the total elapsed time, the total data size in megabytes,
    and the overall processing speed in MB/sec. It logs the total files processed, total data
    processed, and the overall processing speed.
    """

    total_elapsed_time_ns = time.time_ns() - start_time_ns
    total_elapsed_time_s = (
        total_elapsed_time_ns / 1_000_000_000
    )  # Convert nanoseconds to seconds

    total_data_size_mb = total_data_processed / (
        1024 * 1024
    )  # Convert bytes to megabytes

    throughput_pages = total_pages_processed / total_elapsed_time_s  # pages/sec
    throughput_files = total_files / total_elapsed_time_s  # files/sec
    throughput_mb = total_data_size_mb / total_elapsed_time_s  # MB/sec

    logger.info(f"Processed {total_files} files in {total_elapsed_time_s:.2f} seconds.")
    logger.info(f"Total pages processed    : {total_pages_processed:.2f}")
    logger.info(f"Total data processed(MB) : {total_data_size_mb:.2f} MB")
    logger.info(f"Throughput (Pages/sec)   : {throughput_pages:.2f}")
    logger.info(f"Throughput (Files/sec    : {throughput_files:.2f}")
    logger.info(f"Throughput (MB/sec)      : {throughput_mb:.2f}")


def report_statistics(
    start_time_ns,
    stage_elapsed_times,
    total_data_processed,
    total_pages_processed,
    total_files,
):
    """
    Aggregates and reports statistics for the entire processing session.

    Parameters
    ----------
    start_time_ns : int
        The start time of the processing, in nanoseconds.
    stage_elapsed_times : defaultdict(list)
        A defaultdict containing lists of elapsed times for each processing stage.
    total_data_processed : int
        The total amount of data processed, in bytes.
    total_files : int
        The total number of files processed.

    Notes
    -----
    This function first calculates the absolute elapsed time and total trace elapsed time.
    It then delegates to `report_stage_statistics` and `report_overall_speed` to log detailed
    statistics about processing stages and overall processing metrics, respectively.
    """

    abs_elapsed = time.time_ns() - start_time_ns
    total_trace_elapsed = sum(sum(times) for times in stage_elapsed_times.values())
    report_stage_statistics(stage_elapsed_times, total_trace_elapsed, abs_elapsed)
    report_overall_speed(
        total_data_processed, total_pages_processed, start_time_ns, total_files
    )


def determine_processing_function(
    redis_host,
    redis_port,
    task_queue,
    extract,
    extract_method,
    split,
    concurrency_options,
    split_params,
):
    """
    Determines the appropriate processing function and its arguments based on the given
    parameters.

    This function decides between using a direct processing approach or utilizing a distributed
    processing framework based on the provided concurrency options. It configures the processing
    function and its arguments accordingly, ensuring that each source file is processed using the
    specified extraction methods and splitting strategy.

    Parameters
    ----------
    redis_host : str
        The hostname of the Redis server to connect to for task queuing.
    redis_port : int
        The port number of the Redis server.
    task_queue : str
        The name of the Redis task queue where processing tasks should be submitted.
    extract : bool
        A boolean flag indicating whether content extraction tasks should be performed on the
        source files.
    extract_method : list of str
        A list specifying the methods to use for content extraction.
    split : bool
        A boolean flag indicating whether the content of source files should be split according
        to certain criteria.
    concurrency_options : dict
        A dictionary containing options that dictate the concurrency behavior. It includes keys
        such as 'use_dask' to specify whether to use Dask for distributed processing and
        'concurrency_mode' to specify the mode of concurrency (e.g., process-based or
        thread-based).
    split_params : dict
        ...

    Returns
    -------
    tuple
        A tuple where the first element is the selected processing function (callable) and the
        second element is a tuple containing the arguments to be passed to the processing
        function.
        :param split_params:
    """
    use_dask = concurrency_options["use_dask"]
    if use_dask or concurrency_options.get("concurrency_mode") == "process":
        processing_function = process_source
        args = (
            redis_host,
            redis_port,
            task_queue,
            extract,
            extract_method,
            split,
            split_params,
        )
    else:
        # Initialize the RedisClient only if not using Dask or when using thread-based
        # concurrency
        redis_client = RedisClient(host=redis_host, port=redis_port)
        processing_function = _process_source
        args = (redis_client, task_queue, extract, extract_method, split, split_params)

    return processing_function, args


def main(
    file_source,
    redis_host,
    redis_port,
    extract,
    extract_method,
    split,
    dry_run,
    concurrency_options,
    split_params,
    output_directory,
):
    """
    Processes files from specified sources using given extraction and splitting methods,
    and performs actions based on the specified options.

    Parameters
    ----------
    file_source : list
        List of file source paths to be processed.
    redis_host : str
        Hostname of the Redis server.
    redis_port : int
        Port number of the Redis server.
    extract : bool
        Flag indicating whether to perform extraction tasks.
    extract_method : list
        List of methods to use for extraction.
    split : bool
        Flag indicating whether to perform splitting tasks.
    dry_run : bool
        # Not currently implemented
        Flag indicating whether to perform a dry-run, which prints the steps without executing
        them.

    Side Effects
    ------------
    - Files specified in `file_source` are processed according to the `extract` and `split`
      parameters.
    - Progress of file processing is displayed in a tqdm progress bar.
    - Logs the total number of files processed, total data processed, and overall processing
      speed upon completion.

    Notes
    -----
    - This function initializes a Redis client and a ThreadPoolExecutor for concurrent processing.
    - The function calculates and logs the total data processed and the overall processing speed
      in megabytes per second.
    """

    task_queue = os.environ.get("TASK_QUEUE_NAME", "morpheus_task_queue")
    start_time_ns = time.time_ns()
    matching_files = match_and_validate_files(file_source)
    if matching_files is None:
        return

    progress_bar = tqdm(total=len(matching_files), desc="Processing files", unit="file")
    progress_bar.set_postfix(pages_per_sec="0.0")

    with setup_global_executor(**concurrency_options) as executor:
        processing_function, processing_args = determine_processing_function(
            redis_host,
            redis_port,
            task_queue,
            extract,
            extract_method,
            split,
            concurrency_options,
            split_params,
        )

        future_to_file = submit_tasks(
            executor, matching_files, processing_function, processing_args
        )

        (
            stage_elapsed_times,
            total_data_processed,
            total_pages_processed,
        ) = process_tasks(future_to_file, output_directory, progress_bar, start_time_ns)

    progress_bar.close()
    report_statistics(
        start_time_ns,
        stage_elapsed_times,
        total_data_processed,
        total_pages_processed,
        len(matching_files),
    )


@click.command()
@click.option(
    "--file_source",
    multiple=True,
    default=[],
    type=str,
    help="List of file sources/paths to be processed.",
)
@click.option(
    "--dataset_json",
    type=str,
    help="Path to a JSON file containing a list of file sources.",
)
@click.option("--redis_host", default="localhost", help="DNS name for Redis.")
@click.option("--redis_port", default="6379", help="Port for Redis.", type=int)
@click.option("--extract", is_flag=True, help="Enable PDF text extraction task.")
@click.option("--split", is_flag=True, help="Enable text splitting task.")
@click.option(
    "--extract_method",
    default=["pymupdf"],
    type=click.Choice(
        [
            "pymupdf",
            "haystack",
            "tika",
            "unstructured_io",
            "unstructured_service",
            "llama_parse",
        ],
        case_sensitive=False,
    ),
    multiple=True,
    help="Specifies the type(s) of extraction to use.",
)
@click.option(
    "--split_by",
    default="word",
    type=click.Choice(["word", "sentence", "passage"], case_sensitive=False),
    help="Specifies the unit for splitting text: word, sentence, or passage.",
)
@click.option(
    "--split_length",
    default=250,
    type=int,
    help="Specifies the length of each split segment.",
)
@click.option(
    "--split_overlap",
    default=30,
    type=int,
    help="Specifies the overlap between consecutive split segments.",
)
@click.option(
    "--split_max_character_length",
    default=1900,
    type=int,
    help="Specifies the maximum character length of a split segment.",
)
@click.option(
    "--split_sentence_window_size",
    default=0,
    type=int,
    help="Specifies the sentence window size for splitting, if applicable.",
)
@click.option("--use_dask", is_flag=True, help="Use dask for concurrency")
@click.option(
    "--n_workers",
    default=5,
    help="Number of workers for the ThreadPoolExecutor or dask.",
    type=int,
)
@click.option(
    "--log_level",
    default="INFO",
    help="Sets the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
)
@click.option(
    "--concurrency_mode",
    default="thread",
    type=click.Choice(["thread", "process"], case_sensitive=False),
    help="Choose 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Prints the steps to be executed without performing them.",
)
@click.option(
    "--output_directory",
    default=None,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        allow_dash=True,
    ),
    help="Directory where output files will be saved. If provided, must exist and be writable.",
)
def cli(
    file_source,
    dataset_json,
    redis_host,
    redis_port,
    extract,
    extract_method,
    split,
    split_by,
    split_length,
    split_overlap,
    split_max_character_length,
    split_sentence_window_size,
    n_workers,
    log_level,
    concurrency_mode,
    use_dask,
    dry_run,
    output_directory,
):
    """
    CLI entry point for processing files. Configures and executes the main processing function
    based on user inputs.

    The function initializes a global ThreadPoolExecutor and then calls the main processing
    function with the provided options.
    """
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Check if --extract_method is defined but --extract is not specified
    configure_logging(log_level.upper())

    split_params = {
        "split_by": split_by,
        "split_length": split_length,
        "split_overlap": split_overlap,
        "max_character_length": split_max_character_length,
        "sentence_window_size": split_sentence_window_size,
    }

    extract_method = list(extract_method)
    # if a dataset is specified, use it to override the file_source
    if dataset_json:
        with open(dataset_json, "r") as f:
            file_source = json.load(f)

        # Avoid processing files in the same order every time, we don't want to process all pdfs,
        # then txt, etc...
        file_source = file_source["sampled_files"]
        random.shuffle(file_source)

    try:
        concurrency_options = {
            "n_workers": n_workers,
            "concurrency_mode": concurrency_mode,
            "use_dask": use_dask,
        }
        main(
            file_source,
            redis_host,
            redis_port,
            extract,
            extract_method,
            split,
            dry_run,
            concurrency_options,
            split_params,
            output_directory,
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    cli()
