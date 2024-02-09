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
import time

import cudf
import mrc
import redis
from morpheus.messages import ControlMessage
from morpheus._lib.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from pydantic import ValidationError

from morpheus_pdf_ingest.schemas.redis_task_source_schema import RedisTaskSourceSchema
from morpheus_pdf_ingest.util.tracing.latency import ColorCodes, colorize

logger = logging.getLogger(__name__)

RedisTaskSourceLoaderFactory = ModuleLoaderFactory("redis_task_source", "morpheus_pdf_ingest")


@register_module("redis_task_source", "morpheus_pdf_ingest")
def _redis_task_source(builder: mrc.Builder):
    """
    A module for receiving messages from a Redis channel, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """

    module_config = builder.get_current_module_config()
    try:
        validated_config = RedisTaskSourceSchema(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid Redis Task Source configuration: {error_messages}"
        logger.error(log_error_message)

        raise

    # Use validated_config for further operations
    redis_host = validated_config.redis_host
    redis_port = validated_config.redis_port
    task_queue = validated_config.task_queue

    # Initialize Redis client
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

    def fetch_messages() -> ControlMessage:
        """
        Fetch messages from the Redis list (task queue) and yield as ControlMessage.
        """
        while True:
            _, job_payload = redis_client.blpop([task_queue])
            ts_fetched = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

            # Debug Tracing
            ts_entry = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            try:
                job_data = json.loads(job_payload)
                # logger.info(f"job data:\n{json.dumps(job_data, indent=2)}")
                do_trace_tagging = job_data.pop('add_trace_tagging', False)
                tasks = job_data.pop('tasks', [])
                task_id = job_data.pop('task_id')
                ts_send = job_data.pop('latency::ts_send', None)
                data_size_bytes = job_data.pop('data_size_bytes', 0)

                response_channel = f"response_{task_id}"

                df = cudf.DataFrame(job_data.get('data', {}))
                message_meta = MessageMeta(df=df)
                logger.info(f"Received message with {len(df)} rows, cols: {df.columns}")

                control_message = ControlMessage()
                control_message.payload(message_meta)
                control_message.set_metadata('response_channel', response_channel)
                control_message.set_metadata('task_id', task_id)

                for task in tasks:
                    control_message.add_task(task['type'], task['properties'])

                # Debug Tracing
                if do_trace_tagging:
                    ts_exit = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                    control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
                    control_message.set_metadata("trace::entry::redis_task_source", ts_entry)
                    control_message.set_metadata("trace::exit::redis_task_source", ts_exit)

                if (ts_send is not None):
                    elapsed = (ts_fetched - ts_send)
                    elapsed_ms = elapsed / 1e6
                    throughput_mb = (data_size_bytes / (elapsed / 1e9)) / 1e6
                    logger.debug(colorize(f"latency::redis_source_retrieve: {elapsed_ms} msec.",
                                          ColorCodes.BLUE))
                    logger.debug(
                        colorize(f"throughput::redis_source_retrieve: {throughput_mb} MB/sec.",
                                 ColorCodes.BLUE))

                control_message.set_metadata("latency::ts_send", time.clock_gettime_ns(time.CLOCK_MONOTONIC))

                yield control_message
            except Exception as exc:
                logger.error("Error processing message: %s", exc)
                return None

    node = builder.make_source("fetch_messages", fetch_messages)
    builder.register_module_output("output", node)
