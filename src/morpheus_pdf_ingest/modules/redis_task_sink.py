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

import mrc
import redis
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from pydantic import ValidationError

from morpheus_pdf_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from morpheus_pdf_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_sink"
RedisTaskSinkLoaderFactory = ModuleLoaderFactory("redis_task_sink", "morpheus_pdf_ingest")


@register_module(MODULE_NAME, "morpheus_pdf_ingest")
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
    try:
        validated_config = RedisTaskSinkSchema(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid Redis Task Sink configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    # Use validated_config for further operations
    redis_host = validated_config.redis_host
    redis_port = validated_config.redis_port

    # Initialize Redis client
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

    @traceable(MODULE_NAME)
    # @latency_logger(MODULE_NAME)
    def process_and_forward(message: ControlMessage):
        with message.payload().mutable_dataframe() as mdf:
            df_json = mdf.to_json(orient='records')

        # Log the received DataFrame
        # logger.debug(f"\nReceived DataFrame:\n{df}")

        ret_val_json = {
            "data": df_json,
        }

        do_trace_tagging = message.get_metadata('add_trace_tagging', True)
        if (do_trace_tagging):
            traces = {}
            meta_list = message.list_metadata()

            for key in meta_list:
                if (key.startswith("trace::")):
                    traces[key] = message.get_metadata(key)

            ret_val_json["trace"] = traces

        # Send the JSON data to the Redis listener
        response_channel = message.get_metadata('response_channel')

        redis_client.rpush(response_channel, json.dumps(ret_val_json))

        # logger.info(f"Forwarded message to Redis channel '{response_channel}'.")

        return message

    process_node = builder.make_node("process_and_forward", ops.map(process_and_forward))

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
