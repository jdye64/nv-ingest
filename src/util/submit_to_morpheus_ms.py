import asyncio
import base64
import json
import logging
import os
import random
import sys
import uuid

import click
import cudf
import pandas as pd
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


def generate_random_json(num_rows=10):
    """Generates a random JSON data that mimics the structure of Document data."""
    documents = []

    for i in range(num_rows):
        document = {
            "content": random.choice(["Lorem ipsum", "Dolor sit amet", str(random.randint(100, 999))]),
            "content_type": "text",
            "id": f"doc_{i}",
            "meta": {
                "value": random.randint(1, 1000),
                "name": random.choice(["Alice", "Bob", "Charlie"]),
                "is_active": random.choice([True, False])
            }
        }
        documents.append(document)

    return documents


async def submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue, timeout=90):
    job_id = str(uuid.uuid4())
    job_payload = json.dumps({
        "task_id": job_id,
        "add_trace_tagging": True,
        "data": job_data,
        "tasks": tasks
    })
    response_channel = f"response_{job_id}"
    expiration_seconds = 60  # Example: 1 minute

    with open('job_payload.json', 'w') as file:
        file.write(job_payload)

    await redis_client.rpush(task_queue, job_payload)
    await redis_client.expire(response_channel, expiration_seconds)

    print(f"Waiting for response on channel '{response_channel}'...")
    response = await redis_client.blpop(response_channel, timeout=timeout)

    if response:
        _, response_data = response
        await redis_client.delete(response_channel)
        return json.loads(response_data)
    else:
        await redis_client.delete(response_channel)
        raise RuntimeError("No response received within timeout period")


async def extract_file_content(path):
    if path.endswith('.pdf'):
        # For PDF files, read as binary and encode in base64
        with open(path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
            logger.info(f"Encoded PDF content: {content[:100]}... (truncated)")
            logger.info(f"Content length: {len(content)}")

        return content

    raise ValueError(f"Unsupported file type: {path}")


async def load_data_from_path(path):
    print(f"Loading data from paths: {path}")
    result = {"file_name": [], "content": []}
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                content = await extract_file_content(file_path)
                result["file_name"].append(file)
                result["content"].append(content)
    elif os.path.isfile(path):
        content = await extract_file_content(path)
        result["file_name"].append(os.path.basename(path))
        result["content"].append(content)

    return result


@click.command()
@click.option('--file_source', multiple=True, default=[], type=str,
              help='List of file sources/paths to be processed.')
@click.option('--redis-host', default='redis', help="DNS name for Redis.")
@click.option('--redis-port', default='6379', help="Port for Redis.")
def cli(file_source, redis_host, redis_port):
    asyncio.run(main(file_source, redis_host, redis_port))


async def main(file_source, redis_host, redis_port):
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    redis_url = f"redis://{redis_host}:{redis_port}"
    task_queue = os.environ.get("TASK_QUEUE_NAME", "morpheus_task_queue")

    # Create a Redis client instance
    redis_client = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

    if not file_source:
        logger.error("No data source provided.")
        return

    # Prepare job data for each file source
    tasks = [
        {
            "type": "pdf_extract",
            "properties": {
                "extract_text": True,
                "extract_images": False,
                "extract_tables": False
            }
        }
    ]

    for source in file_source:
        try:
            # Assuming load_data_from_paths is modified to work with a single path and return encoded content
            job_data = await load_data_from_path(source),
            response = await submit_job_and_wait_for_response(redis_client, job_data, tasks, task_queue)

            trace_tags = response.get("trace", {})
            df_data = pd.DataFrame.from_dict(json.loads(response.get("data", {})))

            # Assuming multiple documents could be processed
            for content in df_data["content"]:
                logger.info(f"Processed content: {content}")

            # Example of saving a single document's content to a file
            with open(f"{source}_response.json", "w") as file:
                file.write(df_data["content"].iloc[0])

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to process document {source}: {str(e)}")
            # Consider whether to continue processing other files or abort


if __name__ == '__main__':
    cli()
