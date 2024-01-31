import json
import random
import uuid

import cudf
import pyinstrument
import redis


def generate_random_json(num_rows=10):
    """Generates a random JSON array of arrays."""
    data = {
        "value": [random.randint(1, 1000) for _ in range(num_rows)],
        "name": [random.choice(["Alice", "Bob", "Charlie"]) for _ in range(num_rows)],
        "is_active": [random.choice([True, False]) for _ in range(num_rows)]
    }

    return json.dumps(data)


def submit_job_and_wait_for_response(redis_client, job_data, task_queue, timeout=3):
    """Submits a job to a Redis queue and waits for a response."""
    # Generate a unique response queue name for each job
    profiler = pyinstrument.Profiler()
    profiler.start()
    job_id = str(uuid.uuid4())

    # Prepare and send job payload
    job_payload = json.dumps({"job_id": job_id, "data": job_data})
    response_channel = f"response_{job_id}"
    redis_client.rpush(task_queue, job_payload)

    # Wait for response
    print(f"Waiting for response on channel '{response_channel}'...")
    _, response = redis_client.blpop(response_channel, timeout=timeout)
    if not response:
        raise RuntimeError("No response received within timeout period")

    profiler.stop()
    with open('client_profiling.html', 'w') as f:
        f.write(profiler.output_html())

    return cudf.DataFrame.from_dict(json.loads(response))


# Example Usage
redis_host = ('127.0.0.1')
redis_port = 6379
task_queue = 'morpheus_task_queue'
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
random_json = generate_random_json()

response = submit_job_and_wait_for_response(redis_client, random_json, task_queue)
print(f"Response: {response}")
