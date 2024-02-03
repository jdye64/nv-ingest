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
import io
import logging

import cudf
import fitz
import mrc
import mrc.core.operators as ops
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from morpheus_pdf_ingest.schemas.content_extractor_schema import ContentExtractorSchema

logger = logging.getLogger(__name__)

PDFExtractorLoaderFactory = ModuleLoaderFactory("pdf_text_extractor",
                                                "morpheus_pdf_ingest",
                                                ContentExtractorSchema)


def process_pdf_bytes(df):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: cuDF DataFrame with columns 'file_name' and 'content' (base64 encoded PDFs).

    Returns:
    - A cuDF DataFrame with the PDF content replaced by the extracted text.
    """

    # Define a helper function to decode and extract text from a base64 encoded PDF
    def decode_and_extract_text(base64_content):
        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content)
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
    df['content'] = df['content'].apply(decode_and_extract_text)

    return df


@register_module("pdf_text_extractor", "morpheus_pdf_ingest")
def _pdf_text_extractor(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    # try:
    #    extractor_config = ContentExtractorSchema(**module_config)
    # except ValidationError as e:
    #    error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
    #    log_error_message = f"Invalid configuration for file_content_extractor: {error_messages}"
    #    logger.error(log_error_message)
    #    raise ValueError(log_error_message)

    def parse_files(ctrl_msg: ControlMessage) -> ControlMessage:
        df = ctrl_msg.payload().df.to_pandas()
        df = process_pdf_bytes(df)

        ctrl_msg.payload(MessageMeta(df=df))
        return ctrl_msg

    node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
