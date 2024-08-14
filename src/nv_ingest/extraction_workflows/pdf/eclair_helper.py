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

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List
from typing import Tuple

import fitz
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient

from nv_ingest.extraction_workflows.pdf import eclair_utils
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentSubtypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import datetools
from nv_ingest.util.detectors.language import detect_language
from nv_ingest.util.exception_handlers.pdf import pymupdf_exception_handler

logger = logging.getLogger(__name__)

ECLAIR_GRPC_TRITON = os.environ.get("ECLAIR_GRPC_TRITON", "triton:8001")
DEFAULT_BATCH_SIZE = 16
ACCEPTED_CLASSES = set(
    [
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "TOC",
        "Bibliography",
        "Formula",
    ]
)
IGNORED_CLASSES = set(
    [
        "Page-header",
        "Page-footer",
        "Caption",
        "Footnote",
        "Floating-text",
    ]
)


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

    eclair_triton_url = kwargs.get("eclair_grpc_triton", ECLAIR_GRPC_TRITON)

    batch_size = int(kwargs.get("eclair_batch_size", DEFAULT_BATCH_SIZE))

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    identify_nearby_objects = kwargs.get("identify_nearby_objects", True)

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")
    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}

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

        # Convert all pages to fitz.Page.
        pages = []
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            pages.append(page)

        # Split into batches.
        i = 0
        batches = []
        while i < len(pages):
            batches.append(pages[i : i + batch_size])  # noqa: E203
            i += batch_size

        accumulated_text = []
        accumulated_tables = []
        accumulated_images = []

        triton_client = grpcclient.InferenceServerClient(url=eclair_triton_url)

        for batch in batches:
            responses = preprocess_and_send_requests(triton_client, batch)

            for page_idx, raw_text, bbox_offset in responses:
                page_image = None

                classes, bboxes, texts = eclair_utils.extract_classes_bboxes(raw_text)

                page_nearby_blocks = {
                    "text": {"content": [], "bbox": []},
                    "images": {"content": [], "bbox": []},
                    "structured": {"content": [], "bbox": []},
                }

                for cls, bbox, txt in zip(classes, bboxes, texts):
                    if cls in IGNORED_CLASSES:
                        continue

                    elif extract_tables and (cls == "Table"):
                        try:
                            txt = txt.encode().decode("unicode_escape")  # remove double backlashes
                        except UnicodeDecodeError:
                            pass
                        bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                        table = LatexTable(latex=txt, bbox=bbox)
                        accumulated_tables.append(table)

                    elif extract_images and (cls == "Picture"):
                        if page_image is None:
                            page_image, *_ = eclair_utils.pymupdf_page_to_numpy_array(pages[page_idx])

                        base64_img = eclair_utils.crop_image(page_image, bbox)
                        if base64_img:
                            bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                            image = Base64Image(image=base64_img, bbox=bbox)
                            accumulated_images.append(image)

                    elif extract_text and (cls in ACCEPTED_CLASSES):
                        txt = txt.replace("<tbc>", "").strip()  # remove <tbc> tokens (continued paragraphs)
                        txt = eclair_utils.convert_mmd_to_plain_text_ours(txt)

                        if extract_images and identify_nearby_objects:
                            bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                            page_nearby_blocks["text"]["content"].append(txt)
                            page_nearby_blocks["text"]["bbox"].append(bbox)

                        accumulated_text.append(txt)

                # Construct tables
                if extract_tables:
                    for table in accumulated_tables:
                        extracted_data.append(
                            _construct_table_metadata(
                                table,
                                page_idx,
                                page_count,
                                source_metadata,
                                base_unified_metadata,
                            )
                        )
                    accumulated_tables = []

                # Construct images
                if extract_images:
                    for image in accumulated_images:
                        extracted_data.append(
                            _construct_image_metadata(
                                image,
                                page_idx,
                                page_count,
                                source_metadata,
                                base_unified_metadata,
                                page_nearby_blocks,
                            )
                        )
                    accumulated_images = []

                # Construct text - page
                if (extract_text) and (text_depth == TextTypeEnum.PAGE):
                    extracted_data.append(
                        _construct_text_metadata(
                            accumulated_text,
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
                    )
                    accumulated_text = []

        # Construct text - document
        if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
            text_extraction = _construct_text_metadata(
                accumulated_text,
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

        triton_client.close()

    return extracted_data


def preprocess_and_send_requests(
    triton_client,
    batch: List[fitz.Page],
) -> List[Tuple[int, str]]:
    if not batch:
        return []

    page_numbers = []
    page_images = []
    bbox_offsets = []
    for page in batch:
        page_numbers.append(page.number)
        page_image, offset = eclair_utils.pymupdf_page_to_numpy_array(page)
        page_images.append(page_image)
        bbox_offsets.append(offset)

    batch = np.array(page_images)

    input_tensors = [grpcclient.InferInput("image", batch.shape, datatype="UINT8")]
    input_tensors[0].set_data_from_numpy(batch)

    outputs = [grpcclient.InferRequestedOutput("text")]

    query_response = triton_client.infer(
        model_name="eclair",
        inputs=input_tensors,
        outputs=outputs,
    )

    text = query_response.as_numpy("text").tolist()
    text = [t.decode() for t in text]

    if len(text) != len(batch):
        return []

    return list(zip(page_numbers, text, bbox_offsets))


@dataclass
class LatexTable:
    latex: pd.DataFrame
    bbox: Tuple[int, int, int, int]


@dataclass
class Base64Image:
    image: str
    bbox: Tuple[int, int, int, int]


@pymupdf_exception_handler(descriptor="eclair")
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

    extracted_text = "\n\n".join(accumulated_text)

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

    return [ContentTypeEnum.TEXT, validated_unified_metadata.dict(), str(uuid.uuid4())]


@pymupdf_exception_handler(descriptor="eclair")
def _construct_table_metadata(
    table: LatexTable,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    content = table.latex
    table_format = TableFormatEnum.LATEX
    subtype = ContentSubtypeEnum.TABLE
    description = StdContentDescEnum.PDF_TABLE

    content_metadata = {
        "type": ContentTypeEnum.STRUCTURED,
        "description": description,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "line": -1,
            "span": -1,
        },
        "subtype": subtype,
    }
    table_metadata = {
        "caption": "",
        "table_format": table_format,
        "table_location": table.bbox,
    }
    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": content,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "table_metadata": table_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.STRUCTURED, validated_unified_metadata.dict(), str(uuid.uuid4())]


@pymupdf_exception_handler(descriptor="eclair")
def _construct_image_metadata(
    base64_img,
    page_idx,
    page_count,
    source_metadata,
    base_unified_metadata,
    page_nearby_blocks,
):
    content_metadata = {
        "type": ContentTypeEnum.IMAGE,
        "description": StdContentDescEnum.PDF_IMAGE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "line": -1,
            "span": -1,
            "nearby_objects": page_nearby_blocks,
        },
    }

    image_metadata = {
        "image_type": ImageTypeEnum.PNG,
        "structured_image_type": ImageTypeEnum.image_type_1,
        "caption": "",
        "text": "",
        "image_location": base64_img.bbox,
    }

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update(
        {
            "content": base64_img.image,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [ContentTypeEnum.IMAGE, validated_unified_metadata.dict(), str(uuid.uuid4())]
