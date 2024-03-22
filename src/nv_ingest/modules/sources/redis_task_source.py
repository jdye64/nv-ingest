# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import json
import logging
import time
import traceback
from functools import partial

import mrc
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from redis.exceptions import RedisError

import cudf

from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.redis_task_source_schema import RedisTaskSourceSchema
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.redis import RedisClient

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_source"
MODULE_NAMESPACE = "nv_ingest"
RedisTaskSourceLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def fetch_and_process_messages(
    redis_client: RedisClient, validated_config: RedisTaskSourceSchema
):
    """Fetch messages from the Redis list and process them."""

    while True:
        try:
            job_payload = redis_client.fetch_message(validated_config.task_queue)
            ts_fetched = time.time_ns()
            yield process_message(
                job_payload, ts_fetched
            )  # process_message remains unchanged
        except RedisError:
            continue  # Reconnection will be attempted on the next fetch
        except Exception as err:
            logger.error(f"Unexpected error during message processing: {err}")
            traceback.print_exc()


def process_message(job_payload: str, ts_fetched: int) -> ControlMessage:
    """
    Fetch messages from the Redis list (task queue) and yield as ControlMessage.
    """
    ts_entry = time.time_ns()

    job = json.loads(job_payload)
    validate_ingest_job(job)
    job_id = job.pop("job_id")
    job_payload = job.pop("job_payload", {})
    job_tasks = job.pop("tasks", [])

    tracing_options = job.pop("tracing_options", {})
    do_trace_tagging = tracing_options.get("trace", False)
    ts_send = tracing_options.get("ts_send", None)

    response_channel = f"response_{job_id}"

    df = cudf.DataFrame(job_payload)
    message_meta = MessageMeta(df=df)

    control_message = ControlMessage()
    control_message.payload(message_meta)
    control_message.set_metadata("response_channel", response_channel)
    control_message.set_metadata("job_id", job_id)

    for task in job_tasks:
        # logger.debug("Tasks: %s", json.dumps(task, indent=2))
        control_message.add_task(task["type"], task["task_properties"])

    # Debug Tracing
    if do_trace_tagging:
        ts_exit = time.time_ns()
        control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
        control_message.set_metadata(f"trace::entry::{MODULE_NAME}", ts_entry)
        control_message.set_metadata(f"trace::exit::{MODULE_NAME}", ts_exit)

        if ts_send is not None:
            control_message.set_metadata(
                "trace::entry::redis_source_network_in", ts_send
            )
            control_message.set_metadata(
                "trace::exit::redis_source_network_in", ts_fetched
            )

        control_message.set_metadata("latency::ts_send", time.time_ns())

    return control_message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _redis_task_source(builder: mrc.Builder):
    """
    A module for receiving messages from a Redis channel, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """

    validated_config = fetch_and_validate_module_config(builder, RedisTaskSourceSchema)

    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is 0, make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    _fetch_and_process_messages = partial(
        fetch_and_process_messages,
        redis_client=redis_client,
        validated_config=validated_config,
    )

    node = builder.make_source("fetch_messages", _fetch_and_process_messages)
    node.launch_options.engines_per_pe = 6

    builder.register_module_output("output", node)
