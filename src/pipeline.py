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
import time
import typing

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
# from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from morpheus_pdf_ingest.modules.nemo_doc_splitter import NemoDocSplitterLoaderFactory
from morpheus_pdf_ingest.modules.pdf_extractor import PDFExtractorLoaderFactory
from morpheus_pdf_ingest.modules.redis_task_sink import RedisTaskSinkLoaderFactory
from morpheus_pdf_ingest.modules.redis_task_source import RedisTaskSourceLoaderFactory

logger = logging.getLogger(__name__)

CONNECTION_TRACKER = {}

file_sources = [
    './data/pdf_ingest_testing/*.pdf',
]

_source_config = [{
    'type': 'filesystem',
    'name': 'filesystem-test',
    'config': {
        "batch_size": 1,
        "enable_monitor": True,
        "extractor_config": {
            "chunk_size": 512,
            "num_threads": 1,
        },
        "filenames": file_sources,
        "vdb_resource_name": "pdf_ingest_testing"
    }
}]


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
    if ('type' not in source_info or 'name' not in source_info or 'config' not in source_info):
        raise ValueError(f"Each source must have 'type', 'name', and 'config':\n {source_info}")


def setup_nemo_docsplitter_pipe(pipe: Pipeline, config: Config):
    source_module_loader = RedisTaskSourceLoaderFactory.get_instance(module_name="redis_listener",
                                                                     module_config={
                                                                         "redis_listener": {
                                                                             "redis_host": "redis",
                                                                         }
                                                                     })

    source_stage = pipe.add_stage(
        LinearModuleSourceStage(config, source_module_loader, output_type=ControlMessage, output_port_name="output"))

    nemo_splitter_loader = NemoDocSplitterLoaderFactory.get_instance(module_name="nemo_doc_splitter",
                                                                     module_config={
                                                                         "split_by": "word",
                                                                         "split_length": 250,
                                                                         "split_overlap": 30,
                                                                         "max_character_length": 1900,
                                                                     })

    nemo_splitter_stage = pipe.add_stage(
        LinearModulesStage(config, nemo_splitter_loader,
                           input_type=ControlMessage,
                           output_type=ControlMessage,
                           input_port_name="input",
                           output_port_name="output"))

    sink_module_loader = RedisTaskSinkLoaderFactory.get_instance(module_name="redis_task_sink",
                                                                 module_config={
                                                                     "redis_host": "redis",
                                                                 })
    sink_stage = pipe.add_stage(
        LinearModulesStage(config, sink_module_loader,
                           input_type=ControlMessage,
                           output_type=ControlMessage,
                           input_port_name="input",
                           output_port_name="output"))

    pipe.add_edge(source_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, sink_stage)

    return sink_stage


def setup_pdf_ingest_pipe(pipe: Pipeline, config: Config):
    source_module_loader = RedisTaskSourceLoaderFactory.get_instance(module_name="redis_listener",
                                                                     module_config={
                                                                         "redis_listener": {
                                                                             "redis_host": "redis",
                                                                         }
                                                                     })

    source_stage = pipe.add_stage(
        LinearModuleSourceStage(config, source_module_loader, output_type=ControlMessage, output_port_name="output"))

    pdf_text_extract_loader = PDFExtractorLoaderFactory.get_instance(module_name="pdf_extractor",
                                                                     module_config={})

    extractor_stage = pipe.add_stage(
        LinearModulesStage(config, pdf_text_extract_loader,
                           input_type=ControlMessage,
                           output_type=ControlMessage,
                           input_port_name="input",
                           output_port_name="output"))

    nemo_splitter_loader = NemoDocSplitterLoaderFactory.get_instance(module_name="nemo_doc_splitter",
                                                                     module_config={
                                                                         "split_by": "word",
                                                                         "split_length": 60,
                                                                         "split_overlap": 10,
                                                                         "max_character_length": 450,
                                                                     })

    nemo_splitter_stage = pipe.add_stage(
        LinearModulesStage(config, nemo_splitter_loader,
                           input_type=ControlMessage,
                           output_type=ControlMessage,
                           input_port_name="input",
                           output_port_name="output"))

    tokenizer_config = {
        "model_kwargs": {
            "add_special_tokens": False,
            "column": "content",
            "do_lower_case": True,
            "truncation": True,
            "vocab_hash_file": "data/bert-base-uncased-hash.txt",
        },
        "model_name": "bert-base-uncased-hash"
    }
    nlp_stage = pipe.add_stage(PreprocessNLPStage(config, **tokenizer_config.get("model_kwargs", {})))

    sink_module_loader = RedisTaskSinkLoaderFactory.get_instance(module_name="redis_task_sink",
                                                                 module_config={
                                                                     "redis_host": "redis",
                                                                 })

    embeddings_config = {
        "model_kwargs": {
            "force_convert_inputs": True,
            "model_name": "intfloat/e5-small-v2",
            "server_url": "triton:8001",
            "use_shared_memory": True
        }
    }
    embedding_stage = pipe.add_stage(TritonInferenceStage(config, **embeddings_config.get('model_kwargs', {})))

    # vector_db = pipe.add_stage(WriteToVectorDBStage(pipeline_config, **vdb_config))

    sink_stage = pipe.add_stage(
        LinearModulesStage(config, sink_module_loader,
                           input_type=typing.Any,
                           output_type=ControlMessage,
                           input_port_name="input",
                           output_port_name="output"))

    pipe.add_edge(source_stage, extractor_stage)
    pipe.add_edge(extractor_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, nlp_stage)
    pipe.add_edge(nlp_stage, embedding_stage)
    pipe.add_edge(embedding_stage, sink_stage)

    return sink_stage


def pipeline(pipeline_config: Config) -> float:
    logging.info("Starting pipeline setup")

    pipe = Pipeline(pipeline_config)
    start_abs = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # vdb_sources = process_vdb_sources(pipe, pipeline_config, _source_config)
    # cm_sink = pipe.add_stage(InMemorySinkStage(pipeline_config))

    # Create HTTP source
    # setup_nemo_docsplitter_pipe(pipe, pipeline_config)
    setup_pdf_ingest_pipe(pipe, pipeline_config)

    # Connect HTTP source

    # for source_output in vdb_sources:
    #    pipe.add_edge(source_output, cm_sink)

    end_setup = start_run = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    setup_elapsed = (end_setup - start_abs) / 1e9
    logging.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    logging.info("Running pipeline")
    pipe.run()

    end_run = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    run_elapsed = (end_run - start_run) / 1e9
    total_elapsed = (end_run - start_abs) / 1e9

    logging.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logging.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


if (__name__ == "__main__"):
    logging.basicConfig(level=logging.INFO)

    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.pipeline_batch_size = 256
    config.enable_monitor = True
    config.feature_length = 512
    config.model_max_batch_size = 256
    config.mode = PipelineModes.NLP

    pipeline(config)
