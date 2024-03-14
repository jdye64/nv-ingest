# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import os
import time
import typing

from morpheus.config import Config, PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage

from nv_ingest.modules.pdf_extractor import PDFExtractorLoaderFactory
from nv_ingest.stages.pdf_memory_source_stage import PdfMemoryFileSource

logger = logging.getLogger(__name__)


@stage
def no_op_stage(message: typing.Any) -> typing.Any:
    # Return the message for the next stage
    return ["msg"]


def validate_source_config(source_info: typing.Dict[str, any]) -> None:
    """
    Validates the configuration of a source.

    This function checks whether the given source configuration dictionary
    contains all required keys: 'type', 'name', and 'config'.

    Parameters
    ----------
    source_info : typing.Dict[str, any]
        The source configuration dictionary to validate.

    Raises
    ------
    ValueError
        If any of the required keys ('type', 'name', 'config') are missing
        in the source configuration.
    """
    if (
        "type" not in source_info
        or "name" not in source_info
        or "config" not in source_info
    ):
        raise ValueError(
            f"Each source must have 'type', 'name', and 'config':\n {source_info}"
        )


def setup_pdf_ingest_pipe(pipe: Pipeline, config: Config):
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    logger.info(f"REDIS_HOST: {redis_host}")
    logger.info(f"REDIS_PORT: {redis_port}")

    n_pe_workers = 23
    max_queue_size = 1
    dataset_json = "test_output.json"
    delayed_start = False
    repeat_count = 5

    with open(dataset_json, "r") as f:
        source_config = json.load(f)

    source_stage = pipe.add_stage(
        PdfMemoryFileSource(config, source_config, repeat=repeat_count)
    )
    source_monitor = pipe.add_stage(
        MonitorStage(config, description="Source Throughput", unit="msgs")
    )

    trigger_stage = pipe.add_stage(TriggerStage(config))

    pdf_text_extract_loader = PDFExtractorLoaderFactory.get_instance(
        module_name="pdf_extractor",
        module_config={"n_workers": n_pe_workers, "max_queue_size": max_queue_size},
    )

    extractor_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            pdf_text_extract_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    extractor_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Extractor Throughput",
            unit="extractions",
            delayed_start=delayed_start,
        )
    )

    no_op = pipe.add_stage(no_op_stage(config))

    pipeline_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Pipeline Throughput",
            unit="files",
            delayed_start=delayed_start,
        )
    )

    pipe.add_edge(source_stage, source_monitor)
    pipe.add_edge(source_monitor, trigger_stage)
    pipe.add_edge(trigger_stage, extractor_stage)
    pipe.add_edge(extractor_stage, extractor_monitor)
    pipe.add_edge(extractor_monitor, no_op)
    pipe.add_edge(no_op, pipeline_monitor)

    return source_stage


def pipeline(pipeline_config: Config) -> float:
    logging.info("Starting pipeline setup")

    pipe = Pipeline(pipeline_config)
    start_abs = time.time_ns()

    setup_pdf_ingest_pipe(pipe, pipeline_config)

    end_setup = start_run = time.time_ns()
    setup_elapsed = (end_setup - start_abs) / 1e9
    logging.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    logging.info("Running pipeline")
    pipe.run()

    end_run = time.time_ns()
    run_elapsed = (end_run - start_run) / 1e9
    total_elapsed = (end_run - start_abs) / 1e9

    logging.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logging.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.pipeline_batch_size = 256
    config.enable_monitor = True
    config.feature_length = 512
    config.num_threads = os.cpu_count()
    config.model_max_batch_size = 256
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    pipeline(config)
