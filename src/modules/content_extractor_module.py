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

import base64
import logging
import os
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fitz
import fsspec
import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import ValidationError
from pyinstrument import Profiler

from morpheus.messages import MessageMeta
from morpheus.modules.schemas.examples.llm.content_extractor_schema import ContentExtractorSchema
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

ContentExtractorLoaderFactory = ModuleLoaderFactory("file_content_extractor",
                                                    "morpheus_examples_llm",
                                                    ContentExtractorSchema)


@dataclass
class FileMeta:
    file_path: str
    file_name: str
    file_type: str


def get_file_meta(open_file: fsspec.core.OpenFile) -> FileMeta:
    """
    Extract file metadata from the given open file.

    Parameters
    ----------
    open_file: fsspec.core.OpenFile
        OpenFile object

    Returns
    -------
    FileMeta
        Returns FileMeta instance.
    """
    try:
        file_path = open_file.path
        file_name = os.path.basename(file_path)
        _, file_type = os.path.splitext(file_name)

        if len(file_type) > 0:
            file_type = file_type.lstrip('.')
        else:
            file_type = 'none'

        return FileMeta(file_path=file_path, file_name=file_name, file_type=file_type)

    except Exception as e:
        logger.error(f"Error retrieving file metadata for {open_file.path}: {e}")
        raise


has_processed = False


def process_pdf(file_content, file_meta, chunk_size, chunk_overlap):
    """
    Process PDF content using PyMuPDF to extract all text and images, and then split and chunk the text.

    Parameters
    ----------
    file_content : bytes
        The content of the PDF file.
    file_meta : FileMeta
        Metadata about the file.
    chunk_size : int
        Size of each chunk of document.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    list
        A list of dictionaries containing processed data chunks.
    """

    if (not has_processed):
        profiler = Profiler()
        profiler.start()

    processed_data = []

    try:
        # Load PDF from bytes
        pdf_document = fitz.open(stream=file_content, filetype="pdf")

        # Extract and process text and images from all pages
        file_text = ''
        for page in pdf_document:
            # Extract text
            file_text += page.get_text()

            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # xref is the reference number of the image
                base_image = pdf_document.extract_image(xref)
                image_data = base_image["image"]  # Image data in bytes

                encoded_image_data = base64.b64encode(image_data).decode('utf-8')
                # Append image data to processed_data
                processed_data.append({
                    'title': file_meta.file_name,
                    'source': f"{file_meta.file_type}:{file_meta.file_path}",
                    'summary': 'image',
                    'content': encoded_image_data  # binary image data
                })

        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                       length_function=len)
        split_text = text_splitter.split_text(file_text)

        # Chunk and format the text
        for chunk in split_text:
            processed_data.append({
                'title': file_meta.file_name,
                'source': f"{file_meta.file_type}:{file_meta.file_path}",
                'summary': 'text',
                'content': chunk
            })

    except Exception as e:
        logger.error(f"Error processing file {file_meta.file_path} content: {e}")
        return []

    if (not has_processed):
        profiler.stop()
        with open("process_pdf_profile.html", "w") as file:
            file.write(profiler.output_html())

    return processed_data


def read_file(file_path):
    """
    A simple function to read the content of a file and return it.
    This function doesn't do any processing or conversion of the content.

    Parameters
    ----------
    file_path : str
        The path of the file to read.

    Returns
    -------
    str
        The content of the file.
    """
    with open(file_path, 'rb') as file:
        return file.read()


@register_module("file_content_extractor", "morpheus_examples_llm")
def file_content_extractor(builder: mrc.Builder):
    """
    Extracts text from PDF files and constructs a DataFrame with the extracted content.

    This module processes a batch of PDF files, reading their contents and extracting text data to form a DataFrame.
    The module uses a ThreadPoolExecutor for parallel file reading.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'batch_size': int, the number of files to process in parallel.
    - 'num_threads': int, the number of threads to use for parallel file reading.
    - 'chunk_size' : int, size of each chunk of document.
    - 'chunk_overlap' : int, overlap between consecutive chunks.
    - 'converters_meta' : dict, converters configuration.

    The function reads files in parallel but processes the content serially within each batch to prevent CPU contention.

    Example `module_config`
    -----------------------
    {
        "batch_size": 32,
        "num_threads": 10
    }
    """
    module_config = builder.get_current_module_config()

    try:
        extractor_config = ContentExtractorSchema(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration for file_content_extractor: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    batch_size = extractor_config.batch_size
    num_threads = extractor_config.num_threads
    chunk_size = extractor_config.chunk_size
    chunk_overlap = extractor_config.chunk_overlap
    converters_meta = extractor_config.converters_meta

    # Only use PDF converter
    pdf_chunk_params = {
        "chunk_size": converters_meta.get("pdf", {}).get("chunk_size", chunk_size),
        "chunk_overlap": converters_meta.get("pdf", {}).get("chunk_overlap", chunk_overlap)
    }

    def parse_files(open_files: typing.List[fsspec.core.OpenFile]) -> MessageMeta:
        data = []
        _fs = fsspec.filesystem(protocol='file')

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(open_files), batch_size):
                batch = open_files[i:i + batch_size]
                futures = []
                files_meta = []

                for open_file in batch:
                    if not _fs.exists(open_file.path) or _fs.isdir(open_file.path):
                        logger.warning(f"Skipping non-existing or directory file: {open_file.path}")
                        continue

                try:
                    file_meta: FileMeta = get_file_meta(open_file=open_file)
                    if file_meta.file_type == 'pdf':
                        futures.append(executor.submit(read_file, file_meta.file_path))
                        files_meta.append(file_meta)
                    else:
                        logger.warning(f"Skipping non-PDF file: {open_file.path}")

                except Exception as e:
                    logger.error(f"Error processing file {open_file.path}: {e}")

                for file_meta, future in zip(files_meta, futures):
                    if file_meta.file_type == 'pdf':
                        file_content = future.result()
                        if file_content:
                            result = process_pdf(file_content, file_meta, pdf_chunk_params["chunk_size"],
                                                 pdf_chunk_params["chunk_overlap"])
                            if result:
                                data.extend(result)

        df_final = pd.DataFrame(data)

        return MessageMeta(df=df_final)

    node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
