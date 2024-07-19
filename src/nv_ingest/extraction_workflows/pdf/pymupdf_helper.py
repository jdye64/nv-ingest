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
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from math import ceil
from math import floor
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import fitz
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
from PIL import Image

from nv_ingest.extraction_workflows.pdf import yolox_utils
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentSubtypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import bytetools
from nv_ingest.util.converters import datetools
from nv_ingest.util.detectors.language import detect_language
from nv_ingest.util.exception_handlers.pdf import pymupdf_exception_handler

logger = logging.getLogger(__name__)


# Define a helper function to use unstructured-io to extract text from a base64
# encoded bytestram PDF
def pymupdf(pdf_stream, extract_text: bool, extract_images: bool, extract_tables: bool, **kwargs):
    """
    Helper function to use PyMuPDF to extract text from a bytestream PDF.

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

    logger.debug("Extracting PDF with PyMuPDF backend.")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    # TODO(Devin): Not configurable anywhere at the moment; likely don't need to but may be a small perf gain.
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

    if extract_tables:
        extract_tables_method = kwargs.get("extract_tables_method", "yolox")
        if extract_tables_method == "yolox":
            table_detection_endpoint_url = kwargs.get("table_detection_endpoint_url")
            table_detection_model_name = kwargs.get("table_detection_model_name")
            triton_client = grpcclient.InferenceServerClient(url=table_detection_endpoint_url)
            page_images = []

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

        accumulated_text = []
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict["blocks"]  # the list of block dictionaries

            page_nearby_blocks = {
                "text": {"content": [], "bbox": []},
                "images": {"content": [], "bbox": []},
                "structured": {"content": [], "bbox": []},
            }

            for block in blocks:
                block_text = []

                # Extract text (a) - block/line/span
                if (extract_text) and (block["type"] == 0):
                    block_idx = block["number"]
                    for line_idx, line in enumerate(block["lines"]):  # lines a list
                        for span_idx, span in enumerate(line["spans"]):  # spans is a list
                            accumulated_text.append(span["text"])

                            if extract_images and identify_nearby_objects:
                                block_text.append(span["text"])

                            if text_depth == TextTypeEnum.SPAN:
                                text_extraction = _construct_text_metadata(
                                    span,
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
                                )

                                if len(text_extraction) > 0:
                                    extracted_data.append(text_extraction)

                                accumulated_text = []

                        if text_depth == TextTypeEnum.LINE:
                            text_extraction = _construct_text_metadata(
                                line,
                                accumulated_text,
                                keywords,
                                page_idx,
                                block_idx,
                                line_idx,
                                -1,
                                page_count,
                                text_depth,
                                source_metadata,
                                base_unified_metadata,
                            )

                            if len(text_extraction) > 0:
                                extracted_data.append(text_extraction)

                            accumulated_text = []

                    if text_depth == TextTypeEnum.BLOCK:
                        text_extraction = _construct_text_metadata(
                            block,
                            accumulated_text,
                            keywords,
                            page_idx,
                            block_idx,
                            -1,
                            -1,
                            page_count,
                            text_depth,
                            source_metadata,
                            base_unified_metadata,
                        )

                        if len(text_extraction) > 0:
                            extracted_data.append(text_extraction)

                        accumulated_text = []

                if (extract_images and identify_nearby_objects) and (len(block_text) > 0):
                    page_nearby_blocks["text"]["content"].append(" ".join(block_text))
                    page_nearby_blocks["text"]["bbox"].append(block["bbox"])

            if extract_images:
                extracted_image_data = _extract_image_data(
                    page, base_unified_metadata, page_count, page_idx, page_nearby_blocks, source_metadata
                )

                extracted_data.extend(extracted_image_data)

            # Extract text - page (b)
            if (extract_text) and (text_depth == TextTypeEnum.PAGE):
                text_extraction = _construct_text_metadata(
                    page_dict,
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

                if len(text_extraction) > 0:
                    extracted_data.append(text_extraction)

                accumulated_text = []

            # extract page tables
            # currently embedded tables will also part of the accumulated_text
            if extract_tables:
                if extract_tables_method == "yolox":
                    # convert each page to images to prepare for table detection
                    pixmap = page.get_pixmap()
                    page_images.append(
                        np.frombuffer(buffer=pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, -1)
                    )
                else:
                    for table in _extract_tables_using_pymupdf(page):
                        extracted_data.append(
                            _construct_table_or_chart_metadata(
                                table, page_idx, page_count, source_metadata, base_unified_metadata
                            )
                        )

        if extract_tables and extract_tables_method == "yolox":
            for page_idx, table_and_charts in _extract_tables_and_charts_using_table_detection_model(
                page_images,
                triton_client,
                table_detection_model_name,
            ):
                extracted_data.append(
                    _construct_table_or_chart_metadata(
                        table_and_charts, page_idx, page_count, source_metadata, base_unified_metadata
                    )
                )

        # Extract text - document (c)
        if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
            text_extraction = _construct_text_metadata(
                doc,
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

    return extracted_data


@pymupdf_exception_handler(descriptor="pymupdf")
def _construct_text_metadata(
    page_hierarchy_object,
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

    extracted_text = " ".join(accumulated_text)

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

    if text_depth == TextTypeEnum.DOCUMENT:
        bbox = (-1, -1, -1, -1)
    elif text_depth == TextTypeEnum.PAGE:
        width = page_hierarchy_object["width"]
        height = page_hierarchy_object["height"]
        bbox = (0, 0, width, height)
    else:
        bbox = page_hierarchy_object["bbox"]

    text_metadata = {
        "text_type": text_depth,
        "summary": "",
        "keywords": keywords,
        "language": language,
        "text_location": bbox,
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


def _extract_image_data(
    page: fitz.Page, base_unified_metadata, page_count, page_idx, page_nearby_blocks, source_metadata
):
    """
    Extracts image data from a page.

    There are several pitfalls when using pymupdf:
    - Using page.get_images() may extract images from other pages,
      see https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest/-/issues/47
    - Using blocks = page.get_text("dict", sort=True)["blocks"] may not extract all images.
      This is confusing at first, as page.get_image_info() does return all visible images on a page.
      According to the documentation, https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_image_info
      "page.get_image_info() is a subset of the dictionary output of page.get_text()".

    Due to the issues mentioned above, it is imported to use clip=fitz.INFINITE_RECT() argument to be able to extract
    images that may intersect with the pdf page boundaries.
    """
    blocks = page.get_text("dict", sort=True, clip=fitz.INFINITE_RECT())["blocks"]
    image_blocks = [block for block in blocks if block["type"] == 1]
    return [
        _extract_image_from_imageblock(
            block,
            page_idx,
            page_count,
            source_metadata,
            base_unified_metadata,
            page_nearby_blocks,
        )
        for block in image_blocks
    ]


# need to add block text to hierarchy/nearby_objects, including bbox
@pymupdf_exception_handler(descriptor="pymupdf")
def _extract_image_from_imageblock(
    block, page_idx, page_count, source_metadata, base_unified_metadata, page_nearby_blocks
):
    image_type = block["ext"]
    if ImageTypeEnum.has_value(image_type):
        image_type = ImageTypeEnum[image_type.upper()]

    base64_img = bytetools.base64frombytes(block["image"])

    bbox = block["bbox"]
    width = block["width"]
    height = block["height"]
    page_block = block["number"]

    content_metadata = {
        "type": ContentTypeEnum.IMAGE,
        "description": StdContentDescEnum.PDF_IMAGE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": page_block,
            "line": -1,
            "span": -1,
            "nearby_objects": page_nearby_blocks,
        },
    }

    image_metadata = {
        "image_type": image_type,
        "structured_image_type": ImageTypeEnum.image_type_1,
        "caption": "",
        "text": "",
        "image_location": bbox,
        "width": width,
        "height": height,
    }

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update(
        {
            "content": base64_img,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [ContentTypeEnum.IMAGE, validated_unified_metadata.dict(), str(uuid.uuid4())]


@dataclass
class DataFrameTable:
    df: pd.DataFrame
    bbox: Tuple[int, int, int, int]


@dataclass
class ImageTable:
    image: str
    bbox: Tuple[int, int, int, int]


@dataclass
class ImageChart:
    image: str
    bbox: Tuple[int, int, int, int]


def _extract_tables_using_pymupdf(page: fitz.Page) -> List[DataFrameTable]:
    """
    Basic table extraction using PyMuPDF. This function extracts embedded tables from a PDF page.
    """
    # find tables that are marked by vector lines
    # note that horizontal_strategy="text" does not work very well
    tables = page.find_tables(horizontal_strategy="lines", vertical_strategy="lines").tables
    table_dfs = [table.to_pandas() for table in tables]
    # As df is eventually converted to markdown,
    # remove any newlines, tabs, or extra spaces from the column names
    for df in table_dfs:
        df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    bounding_boxes = [table.bbox for table in tables]
    return [DataFrameTable(df, bbox) for df, bbox in zip(table_dfs, bounding_boxes)]


def _extract_tables_and_charts_using_table_detection_model(
    page_images: List[Image.Image],
    triton_client: grpcclient.InferenceServerClient,
    model_name: str,
    batch_size: int = 1,
    num_classes: int = 3,
    conf_thresh: float = 0.48,
    iou_thresh: float = 0.5,
    min_score: float = 0.1,
) -> List[Tuple[int, ImageTable]]:
    tables_and_charts = []
    page_idx = 0

    original_images = [np.array(image) for image in page_images]
    original_image_shapes = [image.shape for image in original_images]

    resized_images = [yolox_utils.resize_image(image, (1024, 1024)) for image in page_images]

    results = []
    batches = [
        np.einsum("bijk->bkij", resized_images[i : i + batch_size]).astype(np.float32)  # noqa: E203
        for i in range(0, len(resized_images), batch_size)
    ]
    for batch in batches:
        input_tensors = [grpcclient.InferInput("input", batch.shape, datatype="FP32")]
        input_tensors[0].set_data_from_numpy(batch)

        outputs = [grpcclient.InferRequestedOutput("output")]

        query_response = triton_client.infer(
            model_name=model_name,
            inputs=input_tensors,
            outputs=outputs,
        )

        output_array = query_response.as_numpy("output")
        pred = yolox_utils.postprocess_model_prediction(
            output_array, num_classes, conf_thresh, iou_thresh, class_agnostic=True
        )
        results += pred

    results = yolox_utils.postprocess_results(results, original_image_shapes, min_score=min_score)
    results = [yolox_utils.expand_chart_bboxes(annotation_dict) for annotation_dict in results]

    for annotation_dict, original_image in zip(results, original_images):
        width, height, *_ = original_image.shape
        for label in ["table", "chart"]:
            objects = annotation_dict[label]
            for idx, bboxes in enumerate(objects):
                *bbox, _ = bboxes
                h1, w1, h2, w2 = bbox * np.array([height, width, height, width])
                cropped = original_image[floor(w1) : ceil(w2), floor(h1) : ceil(h2)]  # noqa: E203
                img = Image.fromarray(cropped.astype(np.uint8))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                base64_img = bytetools.base64frombytes(buffer.getvalue())
                if label == "table":
                    tables_and_charts.append((page_idx, ImageTable(base64_img, (w1, h1, w2, h2))))
                elif label == "chart":
                    tables_and_charts.append((page_idx, ImageChart(base64_img, (w1, h1, w2, h2))))

        page_idx += 1

    return tables_and_charts


@pymupdf_exception_handler(descriptor="pymupdf")
def _construct_table_or_chart_metadata(
    table: Union[DataFrameTable, ImageTable, ImageChart],
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    if isinstance(table, DataFrameTable):
        content = table.df.to_markdown(index=False)
        table_format = TableFormatEnum.MARKDOWN
        subtype = ContentSubtypeEnum.TABLE
        description = StdContentDescEnum.PDF_TABLE
    elif isinstance(table, ImageTable):
        content = table.image
        table_format = TableFormatEnum.IMAGE
        subtype = ContentSubtypeEnum.TABLE
        description = StdContentDescEnum.PDF_TABLE
    elif isinstance(table, ImageChart):
        content = table.image
        table_format = TableFormatEnum.IMAGE
        subtype = ContentSubtypeEnum.CHART
        description = StdContentDescEnum.PDF_CHART
    else:
        raise ValueError("Unknown table type.")

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
