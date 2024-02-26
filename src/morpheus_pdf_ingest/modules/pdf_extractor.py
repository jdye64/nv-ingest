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
import queue
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, Future

import cudf
import pandas as pd
import mrc
import mrc.core.operators as ops
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from morpheus_pdf_ingest.modules import pdf
from morpheus_pdf_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from morpheus_pdf_ingest.util.flow_control import filter_by_task

logger = logging.getLogger(__name__)

MODULE_NAME = "pdf_text_extractor"
PDFExtractorLoaderFactory = ModuleLoaderFactory("pdf_text_extractor",
                                                "morpheus_pdf_ingest",
                                                PDFExtractorSchema)


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
        logger.debug(f"processing ({task_props.get('method', None)}): {df['file_name'][0]}")        
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

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PDF: {e}")

    return extracted_df


@register_module(MODULE_NAME, "morpheus_pdf_ingest")
def _pdf_text_extractor(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    # TODO(Devin): Add schema and make worker pool configurable
    # Global dictionary to store ControlMessages temporarily
    control_messages = {}

    # Initialize a ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=10)
    futures_queue = queue.Queue()

    def forward_func(ctrl_msg: ControlMessage):
        nonlocal control_messages
        nonlocal futures_queue

        # Create a unique task_id for the control message
        task_id = str(uuid.uuid4())

        # Mark the entry time for forwarding
        ts_entry = time.time_ns()
        ctrl_msg.set_metadata("trace::entry::forward", ts_entry)

        # Store the ControlMessage in the global dictionary with the created task_id
        control_messages[task_id] = (ctrl_msg, None)  # No task_props in this case

        # Create a future that is ready to be immediately completed
        future = Future()
        future.set_result(None)  # Indicate that no operation was performed
        future.bypass = True
        future.task_id = task_id
        future.retries = 0  # No retries needed for a forward operation

        # Push the ready future to the futures_queue
        futures_queue.put(future)

    @filter_by_task(["pdf_extract"], forward_func=forward_func)
    def parse_files(ctrl_msg: ControlMessage) -> None:
        ts_entry = time.time_ns()
        ctrl_msg.set_metadata("trace::entry::pdf_extract", ts_entry)
        task_props = ctrl_msg.remove_task('pdf_extract')
        submit_task(ctrl_msg, task_props)

    def submit_task(ctrl_msg, task_props, retry_count=0):
        nonlocal control_messages
        nonlocal futures_queue

        # Implementing backpressure feedback loop
        while futures_queue.qsize() >= 100:  # Queue size cap
            logging.warning("futures_queue is full, waiting for space...")
            time.sleep(1)  # Backoff for 1 second before trying again

        task_id = str(uuid.uuid4())
        df = ctrl_msg.payload().copy_dataframe().to_pandas()

        # Store the ControlMessage in the global dictionary
        # Maybe attach to future instead
        control_messages[task_id] = (ctrl_msg, task_props)

        # Submit the task to the process pool
        future = executor.submit(_process_pdf_bytes, df, task_props)
        future.task_id = task_id
        future.retries = retry_count

        futures_queue.put(future)

    def process_completed_tasks():
        """
        Continuously processes completed tasks. Stops when stop_event is set and the queue is empty.

        Parameters:
        - stop_event: Threading.Event() used to signal when to stop processing.
        """
        nonlocal futures_queue

        while True:
            try:
                # Non-blocking get with timeout to remain responsive to stop_event
                future = futures_queue.get(timeout=0.1)
            except queue.Empty:
                continue  # Queue is empty, loop back to check stop_event and try again

            # Process the completed future
            # TODO(Devin): More error checking
            result = future.result()  # Wait for the future to complete if not already done
            ctrl_msg, task_props = control_messages.pop(future.task_id, None)
            try:
                if ctrl_msg:
                    ts_exit = time.time_ns()
                    if (not hasattr(future, "bypass")):
                        ctrl_msg.payload(MessageMeta(df=cudf.from_pandas(result)))

                    ctrl_msg.set_metadata("trace::exit::pdf_extract", ts_exit)
                    ctrl_msg.set_metadata("latency::ts_send", ts_exit)
                    yield ctrl_msg

            except Exception as e:
                logging.error(f"Error processing task {future.task_id}: {e}")
                retry_count = future.retries
                if (retry_count < 3):  # TODO(Devin) : Make configurable
                    future.retries += 1
                    submit_task(ctrl_msg, task_props, retry_count=future.retries)
                else:
                    logging.error(f"Failed to process task {future.task_id} after {retry_count} retries")
                    ctrl_msg.set_metadata("cm_failed", True)  # TODO(Devin): bring in failure handlers
                    yield ctrl_msg

    # Create a node for processing incoming messages and submitting tasks
    input_node = builder.make_node("pdf_extractor", ops.map(parse_files), ops.filter(lambda x: x is not None))
    builder.register_module_input("input", input_node)

    # Create a source node for handling completed tasks
    completed_task_node = builder.make_source("output_source", process_completed_tasks)
    builder.register_module_output("output", completed_task_node)
