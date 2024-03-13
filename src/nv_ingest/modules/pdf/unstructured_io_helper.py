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

import io
import logging

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

logger = logging.getLogger(__name__)


def unstructured_io(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    **kwargs,
):
    """
    Helper function to use unstructured-io REST API to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether or not to extract text.
    extract_images : bool
        Specifies whether or not to extract images.
    extract_tables : bool
        Specifies whether or not to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.

    Raises
    ------
    SDKError
        If there is an error with the extraction.

    """

    logger.info("Extracting PDF with unstructured-io backend.")

    api_key = kwargs.get("api_key", None)
    unstructured_url = kwargs.get("unstructured_url", None)
    row_data = kwargs.get("row_data", None)
    file_name = row_data["id"] if row_data is not None else "_.pdf"

    s = UnstructuredClient(
        server_url=unstructured_url,
        api_key_auth=api_key,
    )

    files = shared.Files(
        content=pdf_stream,
        file_name=file_name,
    )

    req = shared.PartitionParameters(
        files=files,
        # Other partition params
        strategy="auto",
        languages=["eng"],
    )

    try:
        resp = s.general.partition(req)
        resp_elements = resp.elements

    except SDKError as e:
        logger.error(e)

    text_list = []
    # Extract text from each element of partition response
    for item in resp_elements:
        if item["type"] == "NarrativeText":
            text_list.append(item["text"])

    text = "".join(text_list)

    return text
