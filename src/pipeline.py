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

import logging
import os
import sys
import time
import typing

import click
from morpheus.config import Config, CppConfig, PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage

from nv_ingest.modules.metadata_injector import MetadataInjectorLoaderFactory
from nv_ingest.modules.nemo_doc_splitter import NemoDocSplitterLoaderFactory
from nv_ingest.modules.pdf_extractor import PDFExtractorLoaderFactory
from nv_ingest.modules.redis_task_sink import RedisTaskSinkLoaderFactory
from nv_ingest.modules.redis_task_source import RedisTaskSourceLoaderFactory

logger = logging.getLogger(__name__)


def configure_logging(level_name):
    """
    Configures the global logging level based on a string name.

    Parameters:
    - level_name (str): The name of the logging level (e.g., "DEBUG", "INFO").
    """
    global logger

    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(numeric_level)


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


def setup_pdf_ingest_pipeline(pipe: Pipeline, config: Config):
    # Set proper redis hostname and port
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    logger.info(f"REDIS_HOST: {redis_host}")
    logger.info(f"REDIS_PORT: {redis_port}")

    # Add task-source stage ("redis_listener")
    source_module_loader = RedisTaskSourceLoaderFactory.get_instance(
        module_name="redis_listener",
        module_config={
            "redis_client": {
                "host": redis_host,
                "port": redis_port,
            }
        },
    )
    source_stage = pipe.add_stage(
        LinearModuleSourceStage(
            config,
            source_module_loader,
            output_type=ControlMessage,
            output_port_name="output",
        )
    )

    # Add metadata-injection stage ("pdf_extractor")
    metadata_injector_loader = MetadataInjectorLoaderFactory.get_instance(
        module_name="metadata_injection", module_config={}
    )
    metadata_injector_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            metadata_injector_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add pdf-extraction stage ("pdf_extractor")
    pdf_text_extract_loader = PDFExtractorLoaderFactory.get_instance(
        module_name="pdf_extractor",
        module_config={"n_workers": min(23, os.cpu_count() - 1), "max_queue_size": 1},
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

    # Add doc-splitter stage ("nemo_doc_splitter")
    nemo_splitter_loader = NemoDocSplitterLoaderFactory.get_instance(
        module_name="nemo_doc_splitter",
        module_config={
            "split_by": "word",
            "split_length": 60,
            "split_overlap": 10,
            "max_character_length": 450,
        },
    )
    nemo_splitter_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            nemo_splitter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add task-sink stage ("redis_task_sink")
    sink_module_loader = RedisTaskSinkLoaderFactory.get_instance(
        module_name="redis_task_sink",
        module_config={
            "redis_client": {
                "host": redis_host,
                "port": redis_port,
            }
        },
    )
    sink_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            sink_module_loader,
            input_type=typing.Any,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add edges
    pipe.add_edge(source_stage, metadata_injector_stage)
    pipe.add_edge(metadata_injector_stage, extractor_stage)
    pipe.add_edge(extractor_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, sink_stage)

    return sink_stage


def pipeline(config) -> float:
    logging.info("Starting pipeline setup")

    pipe = Pipeline(config)
    start_abs = time.time_ns()

    setup_pdf_ingest_pipeline(pipe, config)

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


@click.command()
@click.option("--use_cpp", is_flag=True, help="Use C++ backend.")
@click.option(
    "--pipeline_batch_size", default=256, type=int, help="Batch size for the pipeline."
)
@click.option("--enable_monitor", is_flag=True, help="Enable monitoring.")
@click.option("--feature_length", default=512, type=int, help="Feature length.")
@click.option(
    "--num_threads", default=os.cpu_count(), type=int, help="Number of threads."
)
@click.option(
    "--model_max_batch_size", default=256, type=int, help="Model max batch size."
)
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in PipelineModes], case_sensitive=False),
    default=PipelineModes.NLP.value,
    help="Pipeline mode.",
)
@click.option(
    "--log_level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=True
    ),
    default="INFO",
    help="Sets the logging level.",
)
def cli(
    use_cpp,
    pipeline_batch_size,
    enable_monitor,
    feature_length,
    num_threads,
    model_max_batch_size,
    mode,
    log_level,
):
    """
    Command line interface for configuring and running the pipeline with specified options.
    """

    configure_logging(log_level.upper())
    CppConfig.set_should_use_cpp(use_cpp)

    config = Config()
    config.pipeline_batch_size = pipeline_batch_size
    config.enable_monitor = enable_monitor
    config.feature_length = feature_length
    config.num_threads = num_threads
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes[mode.upper()]

    pipeline(config)


if __name__ == "__main__":
    cli()
