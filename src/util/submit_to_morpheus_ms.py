import asyncio
import base64
import json
import logging
import os
import random
import sys
import uuid

import aioredis
import click
import cudf

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


async def submit_job_and_wait_for_response(redis_client, job_data, task_queue, timeout=90):
    job_id = str(uuid.uuid4())
    job_payload = json.dumps({"job_id": job_id, "data": job_data})
    response_channel = f"response_{job_id}"
    expiration_seconds = 60  # Example: 1 minute

    await redis_client.rpush(task_queue, job_payload)
    await redis_client.expire(response_channel, expiration_seconds)

    print(f"Waiting for response on channel '{response_channel}'...")
    response = await redis_client.blpop(response_channel, timeout=timeout)

    if response:
        _, response_data = response
        result = cudf.DataFrame.from_dict(json.loads(response_data))
        await redis_client.delete(response_channel)
        return result
    else:
        await redis_client.delete(response_channel)
        raise RuntimeError("No response received within timeout period")


async def transmit_file_content(path):
    if path.endswith('.pdf'):
        # For PDF files, read as binary and encode in base64
        with open(path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
    else:
        # For other files, assuming they are JSON, load them directly
        with open(path, 'r') as file:
            content = json.load(file)
    return content


async def load_data_from_paths(paths):
    result = {"file_name": [], "content": []}
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    content = await transmit_file_content(file_path)
                    result["file_name"].append(file)
                    result["content"].append(content)
        elif os.path.isfile(path):
            content = await transmit_file_content(path)
            result["file_name"].append(os.path.basename(path))
            result["content"].append(content)

    return result


@click.command()
@click.option('--file_source', multiple=True, default=[], type=str,
              help='List of file sources/paths to be processed.')
@click.option('--redis-host', default='redis', help="DNS name for Redis.")
@click.option('--redis-port', default='6379', help="DNS name for Redis.")
def main(file_source, redis_host, redis_port):
    logger.setLevel(logging.INFO)  # Set the log level to INFO
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)  # stdout handler
    console_handler.setLevel(logging.INFO)  # Set the handler log level to INFO

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    task_queue = 'morpheus_task_queue'

    async def run():
        redis_client = await aioredis.from_url(f"redis://{redis_host}:{redis_port}", encoding="utf-8",
                                               decode_responses=True)

        if (file_source):
            job_data = await load_data_from_paths(file_source)
            logger.debug(json.dumps(job_data, indent=2))
        else:
            raise ValueError("No data source provided.")

        response = await submit_job_and_wait_for_response(redis_client, job_data, task_queue)
        logger.info(f"Received response with {len(response)} rows, cols: {response.columns}")
        with open("response.json", "w") as file:
            file.write(response.to_json(orient='split'))

    asyncio.run(run())


if __name__ == '__main__':
    main()
