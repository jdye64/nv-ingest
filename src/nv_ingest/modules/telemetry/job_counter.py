# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import traceback

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.schemas.job_counter_schema import JobCounterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.telemetry.global_stats import GlobalStats
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "job_counter"
MODULE_NAMESPACE = "nv_ingest"

JobCounterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _job_counter(builder: mrc.Builder) -> None:
    """
    Module for counting submitted jobs and updating the global statistics.

    This module sets up a job counter that increments a specified statistic in the global
    statistics structure each time a message is processed.

    Parameters
    ----------
    builder : mrc.Builder
        The module configuration builder.

    Returns
    -------
    None
    """
    validated_config = fetch_and_validate_module_config(builder, JobCounterSchema)

    stats = GlobalStats.get_instance()

    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def count_jobs(message: ControlMessage) -> ControlMessage:
        try:
            logger.debug(f"Performing job counter: {validated_config.name}")

            stats.increment_stat(validated_config.name)

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to do caption extraction: {e}")

    job_counter_node = builder.make_node(f"{validated_config.name}_counter", ops.map(count_jobs))

    # Register the input and output of the module
    builder.register_module_input("input", job_counter_node)
    builder.register_module_output("output", job_counter_node)
