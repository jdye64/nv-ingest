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
from functools import partial

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from redis import RedisError

from nv_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from nv_ingest.util.converters import dftools
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.redis import RedisClient
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_sink"
MODULE_NAMESPACE = "nv_ingest"

RedisTaskSinkLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def process_and_forward(message: ControlMessage, redis_client: RedisClient):
    """
    Processes the incoming message by converting its payload to JSON and forwarding it to a Redis channel.

    Parameters
    ----------
    message : ControlMessage
        The message containing the data to be processed. It must support payload extraction and metadata operations.
    redis_client : RedisClient
        The Redis client used to forward the processed message. It must support getting a client and pushing messages to
        a channel.

    Returns
    -------
    ControlMessage
        The original message after processing. This is returned as an acknowledgment of processing completion.

    Raises
    ------
    RedisError
        If there is an error while forwarding the processed message to the Redis channel.

    Notes
    -----
    This function first converts the message payload to a JSON string. Then, it checks if trace tagging is enabled and,
    if so, includes trace information in the forwarded message. Finally, the message is sent to the specified Redis
    channel. Errors during message forwarding are logged.
    """
    if message.get_metadata("cm_failed", False):
        logger.error("Received a failed message, skipping processing.")

    with message.payload().mutable_dataframe() as mdf:
        logger.info(f"Received DataFrame with {len(mdf)} rows.")
        # Work around until https://github.com/apache/arrow/pull/40412 is resolved
        df_json = dftools.cudf_to_json(mdf, deserialize_cols=["document_type", "metadata"])

    ret_val_json = {
        "data": df_json,
    }

    do_trace_tagging = message.get_metadata("add_trace_tagging", True)
    if do_trace_tagging:
        traces = {}
        meta_list = message.list_metadata()
        for key in meta_list:
            if key.startswith("trace::"):
                traces[key] = message.get_metadata(key)
        ret_val_json["trace"] = traces

    response_channel = message.get_metadata("response_channel")
    try:
        redis_client.get_client().rpush(response_channel, json.dumps(ret_val_json))
        logger.info(f"Forwarded message to Redis channel '{response_channel}'.")
    except RedisError as e:
        logger.error(f"Failed to forward message to Redis: {e}")

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _redis_task_sink(builder: mrc.Builder):
    """
    Configures and registers a processing node for message handling, including Redis task sinking within a modular
    processing chain.

    Parameters
    ----------
    builder : mrc.Builder
        The modular processing chain builder to which the Redis task sink node will be added. It must support node
        creation and module input/output registration.

    Returns
    -------
    None

    Notes
    -----
    This function initializes a Redis client based on provided configuration, wraps the `process_and_forward` function
    for message processing, and sets up a processing node. It also applies necessary decorators for failure handling
    and trace tagging. The node is then registered as both an input and an output module in the builder, completing the
    setup for message processing and forwarding to Redis.
    """
    validated_config = fetch_and_validate_module_config(builder, RedisTaskSinkSchema)

    # Initialize RedisClient with the validated configuration
    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is always 0 for simplicity, make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    def wrapped_process_and_forward(message: ControlMessage):
        nonlocal redis_client
        func = partial(process_and_forward, redis_client=redis_client)

        return func(message)

    @traceable(MODULE_NAME)
    def _process_and_forward(message: ControlMessage):
        return wrapped_process_and_forward(message)

    process_node = builder.make_node("process_and_forward", ops.map(_process_and_forward))
    process_node.launch_options.engines_per_pe = validated_config.progress_engines

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
