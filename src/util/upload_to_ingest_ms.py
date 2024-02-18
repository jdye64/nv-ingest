import base64
import glob
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import click
import redis
from tqdm import tqdm

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ.get('UNSTRUCTURED_API_KEY', None)
UNSTRUCTURED_URL = os.environ.get('UNSTRUCTURED_URL', "http://localhost:8003")
EXTRACTABLE_FILE_TYPES = ['pdf']

executor = None


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


def setup_global_executor(n_workers, concurrency_mode='thread'):
    """
    Initializes the global ThreadPoolExecutor with the specified number of worker threads.

    Parameters:
    - n_workers (int): The number of worker threads to use in the ThreadPoolExecutor.
    """
    global executor
    executor_class = ThreadPoolExecutor if concurrency_mode == 'thread' else ProcessPoolExecutor
    executor = executor_class(max_workers=n_workers)

    logger.info(f"Global {str(executor_class)} initialized with {n_workers} workers.")


def build_extraction_tasks(methods, file_type):
    """
    Constructs a list of extraction tasks based on specified methods and the file type.

    Parameters
    ----------
    methods : list of str
        The extraction methods to be applied. Possible values might include 'pymupdf',
        'haystack', 'unstructured-local', and 'unstructured-service'.
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
        "extract_images": False,
        "extract_tables": False
    }

    # Define default properties for unstructured tasks
    unstructured_properties = {
        "api_key": UNSTRUCTURED_API_KEY,
        "unstructured_url": UNSTRUCTURED_URL,
    }

    # Add other task types based on _extract_methods
    for method in methods:
        task_props = common_properties.copy()
        task_props['method'] = method
        task = {'type': 'pdf_extract', 'properties': task_props}

        if method == 'pymupdf':
            tasks.append(task)
        elif method in ['haystack', 'unstructured-local', 'unstructured-service']:
            task['properties'].update(unstructured_properties)
        else:
            pass  # Others

        logger.debug('Adding task: %s', json.dumps(task, indent=2))
        tasks.append(task)

    return tasks


def extract_file_content(path):
    """
    Extracts content from a file synchronously. Supports PDF files, which are read as binary and encoded in base64,
    and plain text files, which are read directly.

    Parameters
    ----------
    path : str
        The path to the file from which to extract content.

    Returns
    -------
    str
        The extracted content of the file, base64 encoded for PDFs and plain text for text files.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    if path.endswith('.pdf'):
        # For PDF files, read as binary and encode in base64
        with open(path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
            logger.debug(f"Encoded PDF content: {content[:100]}... (truncated)")
    else:  # Assume everything else is text for now.
        # For plain text files, read as text
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.debug(f"Read plain text content: {content[:100]}... (truncated)")

    logger.debug(f"Content length: {len(content)}")

    return content


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
        content = extract_file_content(file_path)  # Call the synchronous function directly

        return {"file_name": file_name, "id": file_name, "content": content}
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def load_data_from_path(path):
    """
    Loads data from a specified path, supporting both files and directories.
    Utilizes parallel processing to efficiently handle multiple files.

    Parameters
    ----------
    path : str
        The path to a file or directory.

    Returns
    -------
    dict
        A dictionary with keys 'file_name', 'id', and 'content', containing details for each processed file.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the path is neither a file nor a directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    if os.path.isdir(path):
        # Collect all file paths in the directory
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    result = {"file_name": [], "id": [], "content": []}

    future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}

    for future in as_completed(future_to_file):
        file_data = future.result()
        if file_data and 'error' not in file_data:
            result["file_name"].append(file_data["file_name"])
            result["id"].append(file_data["id"])
            result["content"].append(file_data["content"])

    return result


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
    job_payload = json.dumps({
        "add_trace_tagging": True,
        "data": job_data,
        "data_size_bytes": len(json.dumps(job_data)),
        "task_id": job_id,
        "tasks": tasks,
        "latency::ts_send": time.time_ns()
    })
    response_channel = f"response_{job_id}"

    redis_client.rpush(task_queue, job_payload)
    redis_client.expire(response_channel, int(timeout * 1.05))  # Set expiration to timeout+grace period

    logger.debug(f"Waiting for response on channel '{response_channel}'...")
    response = redis_client.blpop(response_channel, timeout=timeout)

    if response:
        _, response_data = response
        redis_client.delete(response_channel)
        return json.loads(response_data)
    else:
        redis_client.delete(response_channel)
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


def process_source(source, redis_client, task_queue, extract, extract_method, split):
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
            "properties": {
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
    job_data = load_data_from_path(source)
    response = submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue, timeout=300)

    return data_size


def main(file_source, redis_host, redis_port, extract, extract_method, split, dry_run):
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
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    task_queue = os.environ.get("TASK_QUEUE_NAME", "morpheus_task_queue")

    start_time_ns = time.time_ns()

    matching_files = list(generate_matching_files(file_source))
    total_files = len(matching_files)

    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")
    total_data_processed = 0  # Total data processed in bytes

    future_to_file = {
        executor.submit(process_source, source, redis_client, task_queue, extract, extract_method, split): source
        for source in matching_files}

    for future in as_completed(future_to_file):
        source = future_to_file[future]
        try:
            data_processed = future.result()
            total_data_processed += data_processed
            progress_bar.update(1)
            # Calculate the average speed up to this point
            current_time_ns = time.time_ns()
            elapsed_time_s = (current_time_ns - start_time_ns) / 1_000_000_000
            average_speed = (total_data_processed / (1024 * 1024)) / elapsed_time_s
            progress_bar.set_postfix_str(f"Avg speed: {average_speed:.2f} MB/s", refresh=True)
        except Exception as e:
            logger.error(f"Error processing file {source}: {str(e)}")

    progress_bar.close()

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
@click.option('--redis-host', default='localhost', help="DNS name for Redis.")
@click.option('--redis-port', default='6379', help="Port for Redis.", type=int)
@click.option('--extract', is_flag=True, help="Enable PDF text extraction task.")
@click.option('--split', is_flag=True, help="Enable text splitting task.")
@click.option('--extract_method',
              type=click.Choice(['pymupdf', 'haystack', 'unstructured_io', 'unstructured_service'],
                                case_sensitive=False), multiple=True,
              help='Specifies the type(s) of extraction to use.')
@click.option('--n_workers', default=5, help="Number of worker threads for the ThreadPoolExecutor.", type=int)
@click.option('--log-level', default='INFO',
              help="Sets the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).")
@click.option('--concurrency_mode', default='thread', type=click.Choice(['thread', 'process'], case_sensitive=False),
              help="Choose 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor.")
@click.option('--dry-run', is_flag=True, help="Prints the steps to be executed without performing them.")
def cli(file_source, redis_host, redis_port, extract, extract_method, split, n_workers, log_level, dry_run,
        concurrency_mode):
    """
    CLI entry point for processing files. Configures and executes the main processing function based on user inputs.

    The function initializes a global ThreadPoolExecutor and then calls the main processing function with the provided options.
    """
    configure_logging(log_level.upper())
    try:
        setup_global_executor(n_workers=n_workers)  # TODO(fix concurrency)
        main(file_source, redis_host, redis_port, extract, extract_method, split, dry_run)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    cli()
