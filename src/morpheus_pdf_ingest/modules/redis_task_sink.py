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

import logging

import mrc
import redis
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

logger = logging.getLogger(__name__)

RedisTaskSinkLoaderFactory = ModuleLoaderFactory("redis_task_sink", "morpheus_pdf_ingest")


@register_module("redis_task_sink", "morpheus_pdf_ingest")
def _redis_task_sink(builder: mrc.Builder):
    """
    A pipeline module that prints message payloads, adds an 'embeddings' column,
    and forwards the payload to a Redis channel.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder to which the pipeline modules will be added.
    """

    module_config = builder.get_current_module_config()
    # Extract Redis configuration from module_config
    redis_host = module_config.get('redis_host', 'redis')
    redis_port = module_config.get('redis_port', 6379)

    # Initialize Redis client
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

    def process_and_forward(message: ControlMessage):
        df = message.payload().df

        # Log the received DataFrame
        logger.info(f"\nReceived DataFrame:\n{df}")

        # Add an 'embeddings' column with example data (adjust as necessary)
        df['embeddings'] = [[0.1, 0.2, 0.3]] * len(df)

        # Convert the DataFrame to JSON
        df_json = df.to_json(orient='records')

        # Send the JSON data to the Redis listener
        response_channel = message.get_metadata('response_channel')
        redis_client.rpush(response_channel, df_json)

        logger.info(f"Forwarded message to Redis channel '{response_channel}'.")

        return message

    process_node = builder.make_node("process_and_forward", ops.map(process_and_forward))

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
