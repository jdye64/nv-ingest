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
import math
import os
import time
import typing

import click
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from pydantic import ValidationError

from nv_ingest.modules.injectors.metadata_injector import MetadataInjectorLoaderFactory
from nv_ingest.modules.sinks.redis_task_sink import RedisTaskSinkLoaderFactory
from nv_ingest.modules.sources.redis_task_source import RedisTaskSourceLoaderFactory
from nv_ingest.modules.transforms.image_caption_extraction import ImageCaptionExtractionLoaderFactory
from nv_ingest.modules.transforms.nemo_doc_splitter import NemoDocSplitterLoaderFactory
from nv_ingest.schemas.ingest_pipeline_config_schema import IngestPipelineConfigSchema
from nv_ingest.stages.docx_extractor_stage import generate_docx_extractor_stage
from nv_ingest.stages.pdf_extractor_stage import generate_pdf_extractor_stage
from nv_ingest.stages.storages.image_storage_stage import ImageStorageStage
from nv_ingest.util.converters.containers import merge_dict
from nv_ingest.util.logging.configuration import LogLevel
from nv_ingest.util.logging.configuration import configure_logging
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def validate_positive(ctx, param, value):
    if value <= 0:
        raise click.BadParameter("must be a positive integer")
    return value


def setup_ingestion_pipeline(pipe: Pipeline, morpheus_pipeline_config: Config, ingest_config):
    # Set proper redis hostname and port
    message_provider_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    message_provider_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")
    logger.info(f"MESSAGE_CLIENT_HOST: {message_provider_host}")
    logger.info(f"MESSAGE_CLIENT_PORT: {message_provider_port}")

    default_cpu_count = math.floor(os.cpu_count() * 0.8)

    # Guard against the requested num_threads being larger than the physical cpu cores available.
    if morpheus_pipeline_config.num_threads:
        # If a configuration value is specified we want to honor it unless it conflicts with available resources
        if os.cpu_count() < morpheus_pipeline_config.num_threads:
            logger.warning(
                "morpheus_pipeline_config.num_threads is set. However, the requested "
                f"{morpheus_pipeline_config.num_threads} CPU cores are not available. "
                f"Defaulting to {default_cpu_count} CPU cores"
            )
        else:
            default_cpu_count = morpheus_pipeline_config.num_threads
    else:
        logger.warn(
            f"morpheus_pipeline_config.num_threads not set. Defaulting to 80% of available CPU cores which is: "
            f"{default_cpu_count}"
        )

    # Add task-source stage ("redis_listener")
    source_module_loader = RedisTaskSourceLoaderFactory.get_instance(
        module_name="redis_listener", module_config=ingest_config.get("redis_task_source", {})
    )
    source_stage = pipe.add_stage(
        LinearModuleSourceStage(
            morpheus_pipeline_config,
            source_module_loader,
            output_type=ControlMessage,
            output_port_name="output",
        )
    )

    metadata_injector_loader = MetadataInjectorLoaderFactory.get_instance(
        module_name="metadata_injection", module_config={}
    )
    metadata_injector_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            metadata_injector_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add pdf extraction stage
    pdf_extractor_config = ingest_config.get("pdf_extractor_module", {})
    pdf_extractor_stage = pipe.add_stage(
        generate_pdf_extractor_stage(
            morpheus_pipeline_config,
            pe_count=pdf_extractor_config.get("n_workers", default_cpu_count),
            task="extract",
            task_desc="pdf_content_extractor",
        )
    )

    # Add docx extraction stage
    docx_extractor_stage = pipe.add_stage(
        generate_docx_extractor_stage(
            morpheus_pipeline_config,
            pe_count=default_cpu_count,
            task="docx-extract",
            task_desc="docx_content_extractor",
        )
    )

    # Add doc-splitter stage ("nemo_doc_splitter")
    nemo_splitter_loader = NemoDocSplitterLoaderFactory.get_instance(
        module_name="nemo_doc_splitter",
        module_config=ingest_config.get("text_splitting_module", {}),
    )
    nemo_splitter_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            nemo_splitter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    image_caption_loader = ImageCaptionExtractionLoaderFactory.get_instance(
        module_name="image_caption_extractor", module_config=ingest_config.get("image_caption_extraction_module", {})
    )
    image_caption_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            image_caption_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add image-storage stage ("image_storage")
    image_storage_stage = pipe.add_stage(ImageStorageStage(morpheus_pipeline_config))

    # Add task-sink stage ("redis_task_sink")
    sink_module_loader = RedisTaskSinkLoaderFactory.get_instance(
        module_name="redis_task_sink",
        module_config=ingest_config.get("redis_task_sink", {}),
    )
    sink_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            sink_module_loader,
            input_type=typing.Any,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    # Add edges
    pipe.add_edge(source_stage, metadata_injector_stage)
    pipe.add_edge(metadata_injector_stage, pdf_extractor_stage)
    pipe.add_edge(pdf_extractor_stage, docx_extractor_stage)
    pipe.add_edge(docx_extractor_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, image_caption_stage)
    pipe.add_edge(image_caption_stage, image_storage_stage)
    pipe.add_edge(image_storage_stage, sink_stage)

    return sink_stage


def pipeline(morpheus_pipeline_config, ingest_config) -> float:
    logging.info("Starting pipeline setup")

    pipe = Pipeline(morpheus_pipeline_config)
    start_abs = time.time_ns()

    setup_ingestion_pipeline(pipe, morpheus_pipeline_config, ingest_config)

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
@click.option(
    "--ingest_config_path", type=str, envvar="NV_INGEST_CONFIG_PATH", help="Path to the JSON configuration file."
)
@click.option("--use_cpp", is_flag=True, help="Use C++ backend.")
@click.option("--pipeline_batch_size", default=256, type=int, help="Batch size for the pipeline.")
@click.option("--enable_monitor", is_flag=True, help="Enable monitoring.")
@click.option("--feature_length", default=512, type=int, help="Feature length.")
@click.option("--num_threads", default=os.cpu_count(), type=int, help="Number of threads.")
@click.option("--model_max_batch_size", default=256, type=int, help="Model max batch size.")
@click.option(
    "--caption_batch_size",
    default=8,
    callback=validate_positive,
    type=int,
    help="Number of captions to process in a batch. Must be a positive integer.",
)
@click.option(
    "--extract_workers",
    default=os.cpu_count(),
    callback=validate_positive,
    type=int,
    help="Number of worker processes for extraction.",
)
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in PipelineModes], case_sensitive=False),
    default=PipelineModes.NLP.value,
    help="Pipeline mode.",
)
@click.option(
    "--log_level",
    type=click.Choice([level.value for level in LogLevel], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Log level.",
)
def cli(
    ingest_config_path,
    caption_batch_size,
    use_cpp,
    pipeline_batch_size,
    enable_monitor,
    extract_workers,
    feature_length,
    num_threads,
    model_max_batch_size,
    mode,
    log_level,
):
    """
    Command line interface for configuring and running the pipeline with specified options.
    """

    configure_logging(logger, log_level.upper())
    CppConfig.set_should_use_cpp(use_cpp)

    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.pipeline_batch_size = pipeline_batch_size
    morpheus_pipeline_config.enable_monitor = enable_monitor
    morpheus_pipeline_config.feature_length = feature_length
    morpheus_pipeline_config.num_threads = num_threads
    morpheus_pipeline_config.model_max_batch_size = model_max_batch_size
    morpheus_pipeline_config.mode = PipelineModes[mode.upper()]

    cli_ingest_config = {}  # TODO(Devin) Create a config for CLI overrides -- not necessary yet.

    if ingest_config_path:
        ingest_config = validate_schema(ingest_config_path)
    else:
        ingest_config = {}

    # Merge command-line options with file configuration
    final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

    # Validate final configuration using Pydantic
    try:
        validated_config = IngestPipelineConfigSchema(**final_ingest_config)
        click.echo(f"Configuration loaded and validated: {validated_config}")
    except ValidationError as e:
        click.echo(f"Validation error: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")
    logger.debug(f"Morpheus configuration:\n{morpheus_pipeline_config}")
    pipeline(morpheus_pipeline_config, final_ingest_config)


if __name__ == "__main__":
    cli()
