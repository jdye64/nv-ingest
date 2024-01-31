# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging

import cudf
import mrc
import redis
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

RedisSubscriberSourceLoaderFactory = ModuleLoaderFactory("redis_listener", "morpheus_pdf_ingest")


@register_module("redis_listener", "morpheus_pdf_ingest")
def _redis_listener(builder: mrc.Builder):
    """
    A module for receiving messages from a Redis channel, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.

    Notes
    -----
    The configuration should contain:
    - 'redis_host': str, the host address of the Redis server.
    - 'redis_port': int, the port on which the Redis server is running.
    - 'task_queue': str, the Redis list from which to retrieve tasks.
    """

    module_config = builder.get_current_module_config()
    redis_config = module_config.get("redis_listener", {})

    # Extract configuration details
    redis_host = redis_config.get('redis_host', 'redis')
    redis_port = redis_config.get('redis_port', 6379)
    task_queue = redis_config.get('task_queue', 'morpheus_task_queue')

    # Initialize Redis client
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

    def fetch_messages() -> ControlMessage:
        """
        Fetch messages from the Redis list (task queue) and yield as ControlMessage.
        """
        while True:
            _, job_payload = redis_client.blpop([task_queue])
            try:
                job_data = json.loads(job_payload)
                df = cudf.read_json(job_data['data'])
                message_meta = MessageMeta(df=df)

                control_message = ControlMessage()
                control_message.payload(message_meta)
                control_message.set_metadata('response_channel', f"response_{job_data['job_id']}")

                yield control_message
            except Exception as exc:
                logger.error("Error processing message: %s", exc)
                logger.error("Message payload: %s", job_payload)

    node = builder.make_source("fetch_messages", fetch_messages)
    builder.register_module_output("output", node)
