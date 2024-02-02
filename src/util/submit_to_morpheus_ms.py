import json
import random
import uuid

import cudf
import pyinstrument
import redis


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


def submit_job_and_wait_for_response(redis_client, job_data, task_queue, timeout=3):
    """Submits a job to a Redis queue, waits for a response with expiration, and cleans up."""
    job_id = str(uuid.uuid4())
    job_payload = json.dumps({"job_id": job_id, "data": job_data})
    response_channel = f"response_{job_id}"

    # Set a reasonable expiration time for the response queue
    # This is a fallback in case the response is never read.
    expiration_seconds = 60  # For example, 1 minute after the response is received

    # Prepare and send job payload
    redis_client.rpush(task_queue, job_payload)

    # Set the expiration time for the response channel
    # Note: We set the expiration now, but since the list is empty, we'll reset it after pushing the response
    redis_client.expire(response_channel, expiration_seconds)

    print(f"Waiting for response on channel '{response_channel}'...")
    _, response = redis_client.blpop(response_channel, timeout=timeout)

    # Reset the expiration time now that we've pushed a response to ensure it lives long enough to be processed
    redis_client.expire(response_channel, expiration_seconds)

    if response:
        # Process the response
        result = cudf.DataFrame.from_dict(json.loads(response))

        # Once the response is processed, delete the response queue to clean up
        redis_client.delete(response_channel)

        return result
    else:
        # Consider cleaning up the response channel if a timeout occurs
        # This is optional and depends on whether you expect late responses to be processed
        redis_client.delete(response_channel)
        raise RuntimeError("No response received within timeout period")



# Example Usage
redis_host = ('127.0.0.1')
redis_port = 6379
task_queue = 'morpheus_task_queue'
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
random_json = generate_random_json()

response = submit_job_and_wait_for_response(redis_client, random_json, task_queue)
print(f"Response: {response}")
