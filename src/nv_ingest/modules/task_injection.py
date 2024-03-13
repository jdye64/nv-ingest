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
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory, register_module
from pydantic import ValidationError

from nv_ingest.schemas.task_injection_schema import TaskInjectionSchema

logger = logging.getLogger(__name__)

TaskExtractionLoaderFactory = ModuleLoaderFactory(
    "task_injection", "nv_ingest", TaskInjectionSchema
)


@register_module("task_injection", "nv_ingest")
def _task_injection(builder: mrc.Builder):
    module_config = builder.get_current_module_config()
    try:
        TaskInjectionSchema(**module_config)
    except ValidationError as e:
        error_messages = "; ".join(
            [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]
        )
        log_error_message = f"Invalid Task Injection configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    def on_data(ctrl_msg: ControlMessage):
        ctrl_msg.get_metadata("task_meta")

        return ctrl_msg

    node = builder.make_node("vdb_resource_tagging", on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
