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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from statistics import mean, median

import chardet
import click
import redis
from nv_ingest.schemas.ingest_job import validate_ingest_job, DocumentTypeEnum
from nv_ingest.util.redis import RedisClient
from tqdm import tqdm

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ.get('UNSTRUCTURED_API_KEY', None)
UNSTRUCTURED_URL = os.environ.get('UNSTRUCTURED_URL', "http://localhost:8003")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", None)
EXTRACTABLE_FILE_TYPES = ['pdf']


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

    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(numeric_level)


def setup_global_executor(n_workers, concurrency_mode='thread', use_dask=False):
    """
    Initializes the global ThreadPoolExecutor with the specified number of worker threads.

    Parameters:
    - n_workers (int): The number of worker threads to use in the ThreadPoolExecutor.
    """

    if use_dask:
        from dask.distributed import Client

        if n_workers == 0:
            n_workers = None  # Use default
        if concurrency_mode == 'thread':
            executor = Client(n_workers=n_workers, threads_per_worker=1, processes=False)
        else:
            executor = Client(n_workers=n_workers, threads_per_worker=1)
        logger.info(f"Global dask client initialized with {n_workers} workers.")
    else:
        executor_class = ThreadPoolExecutor if concurrency_mode == 'thread' else ProcessPoolExecutor
        executor = executor_class(max_workers=n_workers)
        logger.info(f"Global {str(executor_class)} initialized with {n_workers} workers.")

    return executor


def build_extraction_tasks(methods, file_type):
    """
    Constructs a list of extraction tasks based on specified methods and the file type.

    Parameters
    ----------
    methods : list of str
        The extraction methods to be applied. Possible values might include 'pymupdf',
        'haystack', 'unstructured_local', 'unstructured_service', and 'llama_parse'.
    file_type : str
        The type of the file to be processed (e.g., 'pdf', 'txt').

    Returns
    -------
    list of dict
        A list of dictionaries, each representing a task to be executed. Each task dictionary
        includes the type of the task and its properties, such as whether to extract text,
        images, tables, and method-specific configurations.

    Notes
    -----
    - This function dynamically constructs tasks based on the provided methods. If a method
      requires additional properties (like API keys or service URLs for unstructured data
      extraction), these are included in the task's properties.
    - If no valid methods are specified, or if the file type does not require extraction,
      an empty list is returned.
    """
    tasks = []
    if (not methods) or (file_type not in EXTRACTABLE_FILE_TYPES):
        logger.debug("No extraction tasks specified or file type does not require extraction.")
        return tasks

    common_properties = {
        "extract_text": True,
        "extract_images": True,
        "extract_tables": False,
        "text_depth": "document"
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
            'type': 'extract',
            "task_properties": {
                'method': method,
                'document_type': file_type,
                'params': task_props
            }
        }

        if method in ['unstructured-local', 'unstructured-service']:
            task['task_properties']['params'].update(unstructured_properties)
        elif method in ["llama_parse"]:
            task['task_properties']['params'].update(llama_parse_properties)
        else:
            pass  # Others

        logger.debug('Adding task: %s', json.dumps(task, indent=2))
        tasks.append(task)

    return tasks


def extract_file_content(path):
    """
    Extracts content from a file synchronously. Supports PDF files, which are read as binary and encoded in base64,
    and plain text files, which are read with detected encoding.

    Parameters
    ----------
    path : str
        The path to the file from which to extract content.

    Returns
    -------
    str
        The extracted content of the file, base64 encoded for PDFs and in plain text for text files.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    document_type = os.path.basename(path).split('.')[-1].lower()
    if document_type not in DocumentTypeEnum.__members__:  # Check if file type is supported
        raise ValueError(f"Unsupported file type: {document_type}")

    if (document_type != DocumentTypeEnum.txt.name.lower()):
        # For PDF files, read as binary and encode in base64
        with open(path, 'rb') as file:
            encoding = 'utf-8'
            content = base64.b64encode(file.read()).decode(encoding)
            logger.debug(f"Encoded {document_type} content: {content[:100]}... (truncated)")
    else:
        # Detect encoding for non-PDF files
        with open(path, 'rb') as file:
            raw_data = file.read(50000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Re-open the file with the detected encoding
        with open(path, 'r', encoding=encoding) as file:
            content = file.read()
            logger.debug(f"Read plain text content with detected encoding ({encoding}): {content[:100]}... (truncated)")

    logger.debug(f"Content length: {len(content)}")
    return content, document_type


def process_file(file_path):
    """
    Processes a single file by extracting its content and returning file details synchronously.

    Parameters
    ----------
    file_path : str
        The path to the file to be processed.

    Returns
    -------
    dict
        A dictionary containing the file name, id, and extracted content.
    """
    try:
        file_name = os.path.basename(file_path)
        content, document_type = extract_file_content(file_path)  # Call the synchronous function directly

        return {"source_name": file_name, "source_id": file_name, "content": content, "document_type": document_type}
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def load_data_from_path(path):
    """
    Loads data from a specified file path.

    Parameters
    ----------
    path : str
        The path to a file.

    Returns
    -------
    dict
        A dictionary with keys 'file_name', 'id', and 'content', containing details for the processed file.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist or is not a file.
    """

    tasks = []
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
    import copy

    payload = copy.deepcopy(job_payload)
    payload['job_payload']['content'][0] = payload['job_payload']['content'][0][:20] + '...'
    logger.info(json.dumps(payload, indent=2))


def submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue, timeout=90):
    """
    Submits a job to a specified task queue and waits for a response.

    Parameters
    ----------
    redis_client : redis.Redis
        A synchronous Redis client instance.
    job_data : dict
        The job data to be submitted.
    tasks : list
        A list of tasks to be executed as part of the job.
    task_queue : str
        The name of the task queue to submit the job to.
    timeout : int, optional
        The timeout in seconds to wait for a response.

    Returns
    -------
    dict
        The response data as a dictionary.

    Raises
    ------
    RuntimeError
        If no response is received within the timeout period.
    """
    job_id = str(uuid.uuid4())
    job_desc = {
        "job_payload": job_data,
        "job_id": job_id,
        "tasks": tasks,
        "tracing_options": {
            "trace": True,
            "ts_send": time.time_ns()
        }
    }

    # debug_print_job_payload(job_desc)
    validate_ingest_job(job_desc)
    job_payload = json.dumps(job_desc)

    response_channel = f"response_{job_id}"
    response_channel_expiration = int(timeout * 1.05)  # Set expiration to timeout+grace period

    # Use submit_job method of RedisClient
    try:
        response = redis_client.submit_job(task_queue, job_desc, response_channel, response_channel_expiration,
                                           timeout=timeout)
    except:
        traceback.print_exc()
        raise

    if response:
        return response
    else:
        raise RuntimeError("No response received within timeout period")


def generate_matching_files(file_sources):
    """
    Generates files that match the given patterns in file_sources.

    Parameters:
    - file_sources (list): A list of file source patterns.

    Returns:
    - generator: A generator yielding matching file paths.
    """
    files = [file_path for pattern in file_sources for file_path in glob.glob(pattern, recursive=True) if
             os.path.isfile(file_path)]
    for file_path in files:
        yield file_path


def process_source(source, id, redis_host, redis_port, task_queue, extract, extract_method, split):
    # Create RedisClient instead of redis.Redis
    redis_client = RedisClient(host=redis_host, port=redis_port)
    return _process_source(source, id, redis_client, task_queue, extract, extract_method, split)


def _process_source(source, id, redis_client, task_queue, extract, extract_method, split):
    """
    Processes a single source file by applying specified tasks such as splitting and extracting content,
    and submits these tasks along with the job data to a specified Redis task queue.

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
    int
        The size of the data processed from the source file in bytes.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during task preparation or job submission.
    """

    tasks = []
    if split:
        tasks.append({
            "type": "split",
            "task_properties": {
                "split_by": "word",
                "split_length": 250,
                "split_overlap": 30,
                "max_character_length": 1900,
                "sentence_window_size": 0,
            }
        })

    if extract:
        file_type = os.path.basename(source).split('.')[-1].lower()
        extract_tasks = build_extraction_tasks(extract_method, file_type)
        tasks.extend(extract_tasks)

    data_size = os.path.getsize(source)

    try:
        job_data = load_data_from_path(source)
        response = submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue, timeout=300)

        return response, data_size
    except Exception as e:
        return None, 0


def main(file_source, redis_host, redis_port, extract, extract_method, split, dry_run, concurrency_options):
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
        Flag indicating whether to perform a dry-run, which prints the steps without executing them.

    Side Effects
    ------------
    - Files specified in `file_source` are processed according to the `extract` and `split` parameters.
    - Progress of file processing is displayed in a tqdm progress bar.
    - Logs the total number of files processed, total data processed, and overall processing speed upon completion.

    Notes
    -----
    - This function initializes a Redis client and a ThreadPoolExecutor for concurrent processing.
    - The function calculates and logs the total data processed and the overall processing speed in megabytes per second.
    """

    task_queue = os.environ.get("TASK_QUEUE_NAME", "morpheus_task_queue")

    start_time_ns = time.time_ns()

    matching_files = list(generate_matching_files(file_source))
    total_files = len(matching_files)

    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")
    total_data_processed = 0  # Total data processed in bytes

    if not matching_files:
        logger.warning("No files found matching the specified patterns.")
        return

    abs_start = time.time_ns()
    # from dask.distributed import performance_report

    use_dask = concurrency_options["use_dask"]
    with setup_global_executor(**concurrency_options) as executor:
        # with performance_report(filename="dask-report.html"):

        if use_dask or concurrency_options["concurrency_mode"] == "process":
            func = process_source
            args = (redis_host, redis_port, task_queue, extract, extract_method, split)
        else:
            redis_client = RedisClient(host=redis_host, port=redis_port)
            func = _process_source
            args = (redis_client, task_queue, extract, extract_method, split)

        future_to_file = {
            executor.submit(func, source, uuid.uuid4(), *args): source
            for source in matching_files
        }

        stage_elapsed_times = defaultdict(list)

        if use_dask:
            from dask.distributed import as_completed as dask_as_completed

            _as_completed = dask_as_completed
        else:
            _as_completed = as_completed

        for future in _as_completed(future_to_file):
            source = future_to_file[future]
            try:
                response, data_processed = future.result()
                progress_bar.update(1)

                if (response is None):
                    logger.error(f"Error processing file {source}: No response received.")
                    continue

                # Extract trace data from response
                total_data_processed += data_processed
                trace_data = response.get('trace', {})

                # Calculate elapsed time for each stage and store it
                for key, entry_time in trace_data.items():
                    if 'entry' in key:
                        exit_key = key.replace('entry', 'exit')
                        exit_time = trace_data.get(exit_key)
                        if exit_time:
                            stage_name = key.split('::')[2]  # Extract stage name
                            elapsed_time = exit_time - entry_time
                            stage_elapsed_times[stage_name].append(elapsed_time)

                # Calculate the average speed-up to this point
                current_time_ns = time.time_ns()
                elapsed_time_s = (current_time_ns - start_time_ns) / 1_000_000_000
                average_speed = (total_data_processed / (1024 * 1024)) / elapsed_time_s
                progress_bar.set_postfix_str(f"Avg speed: {average_speed:.2f} MB/s", refresh=True)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing file {source}: {str(e)}")

    abs_elapsed = time.time_ns() - abs_start
    progress_bar.close()

    total_trace_elapsed = sum([sum(times) for times in stage_elapsed_times.values()])
    unresolved_time = abs_elapsed - total_trace_elapsed  # Calculate unresolved time

    for stage, times in stage_elapsed_times.items():
        avg_time = mean(times)
        med_time = median(times)
        total_stage_time = sum(times)
        percent_of_total = (total_stage_time / total_trace_elapsed) * 100 if total_trace_elapsed > 0 else 0
        logger.info(
            f"{stage}: Avg: {avg_time / 1e6:.2f} ms, Median: {med_time / 1e6:.2f} ms, Total Time: {total_stage_time / 1e6:.2f} ms, Total % of Trace Computation: {percent_of_total:.2f}%")

    # After iterating through all stages and logging their stats, log the unresolved time
    if total_trace_elapsed > 0 and unresolved_time > 0:
        percent_unresolved = (unresolved_time / abs_elapsed) * 100
        logger.info(
            f"Unresolved time: {unresolved_time / 1e6:.2f} ms, Percent of Total Elapsed: {percent_unresolved:.2f}%")
    elif unresolved_time <= 0:
        logger.info("No unresolved time detected. Trace times account for the entire elapsed duration.")
    else:
        logger.error("Error calculating unresolved time.")

    total_elapsed_time_ns = time.time_ns() - start_time_ns
    total_elapsed_time_s = total_elapsed_time_ns / 1_000_000_000  # Convert nanoseconds to seconds
    total_data_size_mb = total_data_processed / (1024 * 1024)  # Convert bytes to megabytes
    overall_speed = total_data_size_mb / total_elapsed_time_s  # MB/sec

    logger.info(f"Processed {total_files} files in {total_elapsed_time_s:.2f} seconds.")
    logger.info(f"Total data processed: {total_data_size_mb:.2f} MB")
    logger.info(f"Overall processing speed: {overall_speed:.2f} MB/sec")


@click.command()
@click.option('--file_source', multiple=True, default=[], type=str,
              help='List of file sources/paths to be processed.')
@click.option("--dataset_json", type=str, help="Path to a JSON file containing a list of file sources.")
@click.option('--redis-host', default='localhost', help="DNS name for Redis.")
@click.option('--redis-port', default='6379', help="Port for Redis.", type=int)
@click.option('--extract', is_flag=True, help="Enable PDF text extraction task.")
@click.option('--split', is_flag=True, help="Enable text splitting task.")
@click.option('--extract_method', default=['pymupdf'],
              type=click.Choice(['pymupdf', 'haystack', 'tika', 'unstructured_io', 'unstructured_service', 'llama_parse'],
                                case_sensitive=False), multiple=True,
              help='Specifies the type(s) of extraction to use.')
@click.option('--use_dask', is_flag=True, help="Use dask for concurrency")
@click.option('--n_workers', default=5, help="Number of workers for the ThreadPoolExecutor or dask.", type=int)
@click.option('--log_level', default='INFO',
              help="Sets the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).")
@click.option('--concurrency_mode', default='thread', type=click.Choice(['thread', 'process'], case_sensitive=False),
              help="Choose 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor.")
@click.option('--dry-run', is_flag=True, help="Prints the steps to be executed without performing them.")
def cli(file_source, dataset_json, redis_host, redis_port, extract, extract_method, split, n_workers, log_level,
        concurrency_mode, use_dask, dry_run):
    """
    CLI entry point for processing files. Configures and executes the main processing function based on user inputs.

    The function initializes a global ThreadPoolExecutor and then calls the main processing function with the provided options.
    """
    # Check if --extract_method is defined but --extract is not specified
    if extract_method and not extract:
        raise click.UsageError("The --extract option must be specified when using --extract_method.")
    configure_logging(log_level.upper())

    extract_method = list(extract_method)
    # if a dataset is specified, use it to override the file_source
    if dataset_json:
        with open(dataset_json, 'r') as f:
            file_source = json.load(f)

        # Avoid processing files in the same order every time, we don't want to process all pdfs, then txt, etc...
        file_source = file_source['sampled_files']
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
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    cli()
