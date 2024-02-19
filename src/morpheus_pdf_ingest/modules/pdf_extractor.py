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
import uuid
import functools
import io
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor

import cudf
import mrc
import mrc.core.operators as ops
from morpheus._lib.messages import MessageMeta
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from morpheus_pdf_ingest.modules import pdf
from morpheus_pdf_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from morpheus_pdf_ingest.util.flow_control import filter_by_task
from morpheus_pdf_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "pdf_text_extractor"
PDFExtractorLoaderFactory = ModuleLoaderFactory("pdf_text_extractor",
                                                "morpheus_pdf_ingest",
                                                PDFExtractorSchema)

# Global dictionary to store ControlMessages temporarily
control_messages = {}

# Initialize a ProcessPoolExecutor
executor = ProcessPoolExecutor()


def _process_pdf_bytes(df, task_props):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: cuDF DataFrame with columns 'file_name' and 'content' (base64 encoded PDFs).

    Returns:
    - A cuDF DataFrame with the PDF content replaced by the extracted text.
    """

    def decode_and_extract(base64_row, task_props, default="pymupdf"):

        # Base64 content to extract
        base64_content = base64_row["content"]

        # Row data to include in extraction
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props.update({"row_data": row_data})

        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content)

        # Load the PDF
        pdf_stream = io.BytesIO(pdf_bytes)

        # Type of extraction method to use
        extract_method = task_props.get("method", "pymupdf")
        if not hasattr(pdf, extract_method):
            extract_method = default
        try:
            func = getattr(pdf, extract_method, default)
            # logger.info("Running extraction method: %s", extract_method)
            text = func(pdf_stream, **task_props)

            return text
        except Exception as e:
            logger.error(f"Error loading extractor: {e}")

        # TODO: propagate error back and tag message as failed.
        return ""

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(decode_and_extract, task_props=task_props)
        logger.debug(f"Extracting text from PDF: {df['file_name']}")
        # logger.debug(df)
        df['content'] = df.apply(_decode_and_extract, axis=1)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PDF: {e}")

    return df


@register_module(MODULE_NAME, "morpheus_pdf_ingest")
def _pdf_text_extractor(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    @filter_by_task(["pdf_extract"])
    @traceable(MODULE_NAME)
    # @latency_logger(MODULE_NAME)
    def parse_files(ctrl_msg: ControlMessage) -> ControlMessage:
        while (ctrl_msg.has_task('pdf_extract')):
            # get task
            task_props = ctrl_msg.remove_task('pdf_extract')
            with ctrl_msg.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            # Return text, image, or table
            df = _process_pdf_bytes(df, task_props)

            ctrl_msg.payload(MessageMeta(df=cudf.from_pandas(df)))

        return ctrl_msg

    node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
