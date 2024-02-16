import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid

import click
import pandas as pd
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ['UNSTRUCTURED_API_KEY']


async def submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue, timeout=90):
    """
    Submits a job to a specified task queue and waits for a response.

    Parameters
    ----------
    redis_client : aioredis.Redis
        An async Redis client instance.
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

    await redis_client.rpush(task_queue, job_payload)
    await redis_client.expire(response_channel, 60)  # Set expiration to 60 seconds

    logger.info(f"Waiting for response on channel '{response_channel}'...")
    response = await redis_client.blpop(response_channel, timeout=timeout)

    if response:
        _, response_data = response
        await redis_client.delete(response_channel)
        return json.loads(response_data)
    else:
        await redis_client.delete(response_channel)
        raise RuntimeError("No response received within timeout period")


async def extract_file_content(path):
    """
    Asynchronously extracts content from a file. Supports PDF files, which are read as binary and encoded in base64,
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
            logger.info(f"Encoded PDF content: {content[:100]}... (truncated)")
    elif path.endswith('.txt'):
        # For plain text files, read as text
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info(f"Read plain text content: {content[:100]}... (truncated)")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    logger.info(f"Content length: {len(content)}")

    return content


async def process_file(file_path):
    """
    Asynchronously processes a single file by extracting its content and returning file details.

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
        content = await extract_file_content(file_path)
        file_name = os.path.basename(file_path)
        return {"file_name": file_name, "id": file_name, "content": content}
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {"file_name": file_name, "id": file_name, "content": None, "error": str(e)}


async def load_data_from_path(path):
    """
    Asynchronously loads data from a specified path, supporting both files and directories.
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

    tasks = []
    result = {"file_name": [], "id": [], "content": []}

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                tasks.append(process_file(file_path))
    elif os.path.isfile(path):
        tasks.append(process_file(path))
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    files_data = await asyncio.gather(*tasks)

    for file_data in files_data:
        if file_data and 'error' not in file_data:
            result["file_name"].append(file_data["file_name"])
            result["id"].append(file_data["id"])
            result["content"].append(file_data["content"])

    return result


@click.command()
@click.option('--file_source', multiple=True, default=[], type=str,
              help='List of file sources/paths to be processed.')
@click.option('--redis-host', default='localhost', help="DNS name for Redis.")
@click.option('--redis-port', default='6379', help="Port for Redis.")
@click.option('--enable_pdf_extract', is_flag=True, help="Enable PDF text extraction task.")
@click.option('--enable_split', is_flag=True, help="Enable text splitting task.")
def cli(file_source, redis_host, redis_port, enable_pdf_extract, enable_split):
    asyncio.run(main(file_source, redis_host, redis_port, enable_pdf_extract, enable_split))


async def main(file_source, redis_host, redis_port, enable_pdf_extract, enable_split):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    redis_url = f"redis://{redis_host}:{redis_port}"
    task_queue = os.environ.get("TASK_QUEUE_NAME", "morpheus_task_queue")

    redis_client = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

    if not file_source:
        logger.error("No data source provided.")
        return

    tasks = []
    if enable_pdf_extract:
        tasks.append({
            "type": "pdf_extract",
            "properties": {
                "type": "haystack",
                "api_key": UNSTRUCTURED_API_KEY,
                "unstructured_url": "http://localhost:8003",
                "extract_text": True,
                "extract_images": False,
                "extract_tables": False
            }
        })

    if enable_split:
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

    # The rest of your main function goes here, which processes each file source.

    for source in file_source:
        try:
            # Assuming load_data_from_paths is modified to work with a single path and return encoded content
            job_data = await load_data_from_path(source)
            response = await submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue,
                                                              timeout=300)

            trace_tags = response.get("trace", {})
            df_data = pd.DataFrame.from_dict(json.loads(response.get("data", {})))

            # Assuming multiple documents could be processed
            logger.info(f"Received {len(df_data['content'])} documents from source: {source}")

            # Example of saving a single document's content to a file
            with open(f"{source}_response.json", "w") as file:
                file.write(df_data["content"].iloc[0])
            with open(f"{source}_response_data.csv", "w") as file:
                df_data.to_csv(file)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to process document {source}: {str(e)}")
            # Consider whether to continue processing other files or abort


if __name__ == '__main__':
    cli()
