# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

import json
import logging
import uuid
from datetime import datetime

import fitz
import numpy as np
import tritonclient.grpc as grpcclient

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import datetools
from nv_ingest.util.detectors.language import detect_language
from nv_ingest.util.exception_handlers.pdf import pymupdf_exception_handler

logger = logging.getLogger(__name__)

DEFAULT_ECLAIR_TRITON_HOST = "localhost"
DEFAULT_ECLAIR_TRITON_PORT = 8001
DEFAULT_DPI = 96
MAX_BATCH_SIZE = 4


# Define a helper function to use Eclair to extract text from a base64 encoded bytestram PDF
def eclair(pdf_stream, extract_text: bool, extract_images: bool, extract_tables: bool, **kwargs):
    """
    Helper function to use Eclair to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.
    """
    logger.debug("Extracting PDF with Eclair backend.")

    eclair_triton_host = kwargs.get("eclair_triton_host", DEFAULT_ECLAIR_TRITON_HOST)
    eclair_triton_port = kwargs.get("eclair_triton_port", DEFAULT_ECLAIR_TRITON_PORT)
    eclair_triton_url = f"{eclair_triton_host}:{eclair_triton_port}"
    triton_client = grpcclient.InferenceServerClient(url=eclair_triton_url)

    batch_size = kwargs.get("batch_size", MAX_BATCH_SIZE)

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]
    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")
    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    base_unified_metadata = json.loads(row_data[metadata_col]) if metadata_col in row_data.index else {}

    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    # each row is a partition level document
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        extracted_data = []

        page_count = doc.page_count
        filename = doc.name

        # last_modified
        last_modified = doc.metadata.get("modDate", None)
        if last_modified in (
            None,
            "",
        ):
            last_modified = datetools.remove_tz(datetime.now()).isoformat()
        else:
            last_modified = datetools.datetimefrompdfmeta(last_modified)

        # date_created
        date_created = doc.metadata.get("creationDate", None)
        if date_created in (
            None,
            "",
        ):
            date_created = datetools.remove_tz(datetime.now()).isoformat()
        else:
            date_created = datetools.datetimefrompdfmeta(date_created)

        # keywords
        keywords = doc.metadata.get("keywords", [])
        # source_type
        source_type = doc.metadata.get("format", SourceTypeEnum.PDF)

        source_metadata = {
            "source_name": filename,
            "source_id": source_id,
            "source_location": source_location,
            "source_type": source_type,
            "collection_id": collection_id,
            "date_created": date_created,
            "last_modified": last_modified,
            "summary": "",
            "partition_id": partition_id,
            "access_level": access_level,
        }

        image_arrays = []
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            # Extract text - page
            if extract_text:
                pixmap = page.get_pixmap(dpi=DEFAULT_DPI)
                image_arrays.append(
                    np.frombuffer(buffer=pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, -1)
                )

            # extract page tables
            if extract_tables:
                pass

        batches = [
            np.array(image_arrays[i : i + batch_size]) for i in range(0, len(image_arrays), batch_size)  # noqa:  E203
        ]

        extracted_text = []
        for batch in batches:
            input_tensors = [grpcclient.InferInput("image", batch.shape, datatype="UINT8")]
            input_tensors[0].set_data_from_numpy(batch)

            outputs = [grpcclient.InferRequestedOutput("text")]

            query_response = triton_client.infer(
                model_name="eclair",
                inputs=input_tensors,
                outputs=outputs,
            )

            text = query_response.as_numpy("text").tolist()
            extracted_text.extend([t.decode("utf-8") for t in text])

        # Construct text - page
        if (extract_text) and (text_depth == TextTypeEnum.PAGE):
            for page_idx, page_text in enumerate(extracted_text):
                text_extraction = _construct_text_metadata(
                    page_text,
                    keywords,
                    page_idx,
                    -1,
                    -1,
                    -1,
                    page_count,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                )

                if len(text_extraction) > 0:
                    extracted_data.append(text_extraction)

        # Construct text - document
        if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
            extracted_text = "".join(extracted_text)

            text_extraction = _construct_text_metadata(
                extracted_text,
                keywords,
                -1,
                -1,
                -1,
                -1,
                page_count,
                text_depth,
                source_metadata,
                base_unified_metadata,
            )

            if len(text_extraction) > 0:
                extracted_data.append(text_extraction)

    return extracted_data


@pymupdf_exception_handler(descriptor="pymupdf")
def _construct_text_metadata(
    accumulated_text,
    keywords,
    page_idx,
    block_idx,
    line_idx,
    span_idx,
    page_count,
    text_depth,
    source_metadata,
    base_unified_metadata,
):
    if len(accumulated_text) < 1:
        return []

    extracted_text = "".join(accumulated_text)

    content_metadata = {
        "type": ContentTypeEnum.TEXT,
        "description": StdContentDescEnum.PDF_TEXT,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": block_idx,
            "line": line_idx,
            "span": span_idx,
        },
    }

    language = detect_language(extracted_text)

    text_metadata = {
        "text_type": text_depth,
        "summary": "",
        "keywords": keywords,
        "language": language,
    }

    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": extracted_text,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "text_metadata": text_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    return [ContentTypeEnum.TEXT.value, validated_unified_metadata.dict(), str(uuid.uuid4())]
