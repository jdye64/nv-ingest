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
import functools
import io
import logging

import fitz
import mrc
import mrc.core.operators as ops
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from morpheus_pdf_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from morpheus_pdf_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "pdf_text_extractor"
PDFExtractorLoaderFactory = ModuleLoaderFactory("pdf_text_extractor",
                                                "morpheus_pdf_ingest",
                                                PDFExtractorSchema)


def _process_pdf_bytes(df, extract_text: bool = False, extract_images: bool = False,
                       extract_tables: bool = False):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: cuDF DataFrame with columns 'file_name' and 'content' (base64 encoded PDFs).

    Returns:
    - A cuDF DataFrame with the PDF content replaced by the extracted text.
    """

    # Define a helper function to decode and extract text from a base64 encoded PDF
    def decode_and_extract(base64_content, extract_text: bool, extract_images: bool,
                           extract_tables: bool):
        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content[0])
        # Load the PDF
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        # Extract text from each page
        for page in doc:
            text += page.get_text()
        doc.close()  # Close the document
        return text

    # Apply the helper function to each row in the 'content' column
    _decode_and_extract = functools.partial(decode_and_extract, extract_text=extract_text,
                                            extract_images=extract_images, extract_tables=extract_tables)
    logger.info(f"Extracting text from PDFs: {df['file_name']}")
    df['content'] = df['content'].apply(_decode_and_extract)

    return df


@register_module(MODULE_NAME, "morpheus_pdf_ingest")
def _pdf_text_extractor(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    @traceable(MODULE_NAME)
    def parse_files(ctrl_msg: ControlMessage) -> ControlMessage:
        while (ctrl_msg.has_task('pdf_extract')):
            # get task
            task = ctrl_msg.remove_task('pdf_extract')
            task_props = task.get('properties', {})
            extract_text = task_props.get('extract_text', False)
            extract_images = task_props.get('extract_images', False)
            extract_tables = task_props.get('extract_tables', False)

            df = ctrl_msg.payload().df.to_pandas()

            # Return text, image, or table
            df = _process_pdf_bytes(df, extract_text, extract_images, extract_tables)

            ctrl_msg.payload(MessageMeta(df=df))

        return ctrl_msg

    node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
