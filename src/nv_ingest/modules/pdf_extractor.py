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
import traceback

import cudf
import mrc
import mrc.core.operators as ops
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from nv_ingest.modules import pdf
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.exception_handlers.pdf import create_exception_tag
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "pdf_content_extractor"
MODULE_NAMESPACE = "nv-ingest"
PDFExtractorLoaderFactory = ModuleLoaderFactory(MODULE_NAME,
                                                MODULE_NAMESPACE,
                                                PDFExtractorSchema)


def _process_pdf_bytes(df, task_props):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: cuDF DataFrame with columns 'source_id' and 'content' (base64 encoded PDFs).

    Returns:
    - A cuDF DataFrame with the PDF content replaced by the extracted text.
    """

    def decode_and_extract(base64_row, task_props, default="pymupdf"):

        # Base64 content to extract
        base64_content = base64_row["content"]
        # Row data to include in extraction
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data
        # Get source_id
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None
        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content)

        # Load the PDF
        pdf_stream = io.BytesIO(pdf_bytes)

        # Type of extraction method to use
        extract_method = task_props.get("method", "pymupdf")
        extract_params = task_props.get("params", {})
        if not hasattr(pdf, extract_method):
            extract_method = default
        try:
            func = getattr(pdf, extract_method, default)
            logger.info("Running extraction method: %s", extract_method)
            extracted_data = func(pdf_stream, **extract_params)

            return extracted_data

        except Exception as e:
            traceback.print_exc()
            log_error_message = f"Error loading extractor:{e}"
            logger.error(log_error_message)
            logger.error(f"Failed on file:{source_id}")           

        # Propagate error back and tag message as failed.
        exception_tag = create_exception_tag(
            error_message=log_error_message,
            source_id=source_id)

        return exception_tag

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(decode_and_extract, task_props=task_props)
        logger.debug(f"processing ({task_props.get('method', None)})")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(
                sr_extraction.to_list(),
                columns=[
                    'document_type',
                    'metadata'])
        else:
            return pd.DataFrame(columns=["document_type", "metadata"])

        return extracted_df

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PDF: {e}")

    return df


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _pdf_text_extractor(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    @filter_by_task(["extract"])
    @traceable(MODULE_NAME)
    def process_task(ctrl_msg: ControlMessage) -> None:
        task_props = ctrl_msg.remove_task('extract')

        df = ctrl_msg.payload().copy_dataframe().to_pandas()
        try:
            result = _process_pdf_bytes(df, task_props)
            df_result = cudf.DataFrame(result, columns=["document_type", "metadata"])
            ctrl_msg.payload(MessageMeta(df=df_result))
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            ctrl_msg.set_metadata("cm_failed", True)

        return ctrl_msg

    # Create a node for processing incoming messages and submitting tasks
    input_node = builder.make_node("pdf_content_extractor", ops.map(process_task),
                                   ops.filter(lambda x: x is not None))
    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
