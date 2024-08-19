# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import io
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from math import floor
from math import log
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
            cached_nim_endpoint_url = kwargs.get("cached_nim_endpoint_url", "cached:8001")
            deplot_nim_endpoint_url = kwargs.get("deplot_nim_endpoint_url", "deplot:8001")
            paddle_nim_endpoint_url = kwargs.get("paddle_nim_endpoint_url", "paddle:8001")
            pages = []

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
                    pages.append(page)
                else:
                    for table in _extract_tables_using_pymupdf(page):
                        extracted_data.append(
                            _construct_table_or_chart_metadata(
                                table, page_idx, page_count, source_metadata, base_unified_metadata
                            )
                        )

        if extract_tables and extract_tables_method == "yolox":
            for page_idx, table_and_charts in extract_tables_and_charts_using_image_ensemble(
                pages,
                table_detection_endpoint_url,
                table_detection_model_name,
                cached_nim_endpoint_url,
                deplot_nim_endpoint_url,
                paddle_nim_endpoint_url,
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
    """
    Construct text metadata for extracted text from a PDF page.

    This function processes accumulated text extracted from a PDF page and generates structured metadata. The metadata
    includes information about the text's location within the document, its language, and hierarchical position
    (e.g., page, block, line, span). The function also accounts for various levels of text granularity
    (document, page, or specific text block) and validates the resulting metadata structure.

    Parameters
    ----------
    page_hierarchy_object : dict
        A dictionary containing details about the current page or text block hierarchy, including dimensions and
        bounding box information.
    accumulated_text : list of str
        A list of text strings accumulated from the PDF, representing the content to be processed.
    keywords : list of str
        A list of keywords associated with the extracted text, used for categorization or search purposes.
    page_idx : int
        The index of the current page (0-based) from which the text is extracted.
    block_idx : int
        The index of the block on the page containing the text.
    line_idx : int
        The index of the line within the block containing the text.
    span_idx : int
        The index of the span within the line containing the text.
    page_count : int
        The total number of pages in the document.
    text_depth : TextTypeEnum
        An enumeration value indicating the depth or granularity of the text being processed (e.g., document-level,
        page-level, or block-level).
    source_metadata : dict
        Metadata related to the source document from which the text is extracted.
    base_unified_metadata : dict
        A base dictionary containing pre-existing metadata that needs to be merged with the text-specific metadata.

    Returns
    -------
    list
        A list containing three elements:
        - ContentTypeEnum.TEXT: The type of content extracted, which is text.
        - dict: A dictionary of the validated and structured unified metadata.
        - str: A unique identifier (UUID) for the text content.

    Notes
    -----
    - The function only proceeds if `accumulated_text` contains at least one element; otherwise, it returns an empty
    list.
    - Text is accumulated into a single string and metadata is constructed to describe the text's location, type, and
    associated language.
    - The bounding box (`bbox`) for the text is determined based on the `text_depth`. If the depth is at the document
    level, a default bounding box of `(-1, -1, -1, -1)` is used. If the depth is at the page level, the bounding box
    covers the entire page. Otherwise, the bounding box is taken from the `page_hierarchy_object`.
    - The `pymupdf_exception_handler` decorator is applied to handle any exceptions related to PyMuPDF operations.

    Examples
    --------
    >>> page_hierarchy_object = {"bbox": (100, 200, 300, 400), "width": 600, "height": 800}
    >>> accumulated_text = ["This is some extracted text."]
    >>> keywords = ["example", "text"]
    >>> page_idx = 0
    >>> block_idx = 1
    >>> line_idx = 2
    >>> span_idx = 3
    >>> page_count = 10
    >>> text_depth = TextTypeEnum.BLOCK
    >>> source_metadata = {"source": "document_source"}
    >>> base_unified_metadata = {"base_key": "base_value"}
    >>> result = _construct_text_metadata(
    >>>     page_hierarchy_object, accumulated_text, keywords, page_idx, block_idx,
    >>>     line_idx, span_idx, page_count, text_depth, source_metadata, base_unified_metadata
    >>> )
    >>> print(result)
    [ContentTypeEnum.TEXT, {...}, 'a1b2c3d4-5678-90ab-cdef-1234567890ab']

    References
    ----------
    - PyMuPDF documentation: https://pymupdf.readthedocs.io/en/latest/
    """
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
    Extract image data from a PDF page using PyMuPDF.

    This function processes a PDF page to extract image data, addressing several known issues with PyMuPDF's
    image extraction methods. Specifically, it mitigates problems where images from other pages might be
    incorrectly extracted, or where not all images are detected when using certain methods.

    There are several pitfalls when using PyMuPDF:
    - Using `page.get_images()` may extract images from other pages.
      For more details, see: https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest/-/issues/47
    - Using `blocks = page.get_text("dict", sort=True)["blocks"]` may not extract all images.
      This can be confusing, as `page.get_image_info()` does return all visible images on a page.
      According to the documentation: https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_image_info,
      "page.get_image_info() is a subset of the dictionary output of `page.get_text()`".

    Due to the issues mentioned above, it is important to use the `clip=fitz.INFINITE_RECT()` argument to
    ensure that images intersecting with the PDF page boundaries are correctly extracted.

    Parameters
    ----------
    page : fitz.Page
        The page object from which images are to be extracted.
    base_unified_metadata : dict
        A base dictionary containing pre-existing metadata that will be merged with the image-specific metadata.
    page_count : int
        The total number of pages in the PDF document.
    page_idx : int
        The index of the current page (0-based) from which images are being extracted.
    page_nearby_blocks : list
        A list of nearby text blocks on the page, used for providing context or hierarchical structure in the metadata.
    source_metadata : dict
        Metadata associated with the source document from which the images are extracted.

    Returns
    -------
    list
        A list of extracted image data, where each entry is the result of the `_extract_image_from_imageblock` function.
        Each entry in the list contains the image type, structured metadata, and a unique identifier (UUID).

    Notes
    -----
    - The function uses `page.get_text("dict", sort=True, clip=fitz.INFINITE_RECT())` to retrieve a dictionary of
      blocks that may contain images.
    - The `clip=fitz.INFINITE_RECT()` parameter ensures that even images that intersect with the page boundaries are
    included.
    - The blocks are filtered to include only those where `block["type"] == 1`, which indicates an image block.
    - Each image block is then processed using the `_extract_image_from_imageblock` function to extract the image
      and its associated metadata.

    Examples
    --------
    >>> page = fitz.open("sample.pdf").load_page(0)
    >>> base_unified_metadata = {"author": "Author Name"}
    >>> page_count = 10
    >>> page_idx = 0
    >>> page_nearby_blocks = ["Text block 1", "Text block 2"]
    >>> source_metadata = {"source": "source_info"}
    >>> images = _extract_image_data(page, base_unified_metadata, page_count, page_idx, page_nearby_blocks,
    source_metadata)
    >>> for image in images:
    >>>     print(image)

    References
    ----------
    - PyMuPDF documentation: https://pymupdf.readthedocs.io/en/latest/page.html
    - Issue regarding `page.get_images()`:
    https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest/-/issues/47
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
    """
    Extracts image data from an image block on a PDF page and structures the metadata.

    This function processes an image block extracted from a PDF page using PyMuPDF and structures the image data
    and metadata in a standardized format. The image is converted to a base64-encoded string, and metadata about
    the image and its location on the page is collected and validated.

    Parameters
    ----------
    block : dict
        A dictionary containing the image block data, including the image itself, its bounding box,
        and other relevant properties.
    page_idx : int
        The index of the page from which the image is extracted.
    page_count : int
        The total number of pages in the document.
    source_metadata : dict
        Metadata associated with the source document from which the image is extracted.
    base_unified_metadata : dict
        A base dictionary containing pre-existing metadata that needs to be merged with the image-specific metadata.
    page_nearby_blocks : list
        A list of nearby blocks on the page, used for providing context or hierarchical structure in the metadata.

    Returns
    -------
    list
        A list containing three elements:
        - ContentTypeEnum.IMAGE: The type of content extracted, which is an image.
        - dict: A dictionary of the validated and structured unified metadata.
        - str: A unique identifier (UUID) for the image.

    Notes
    -----
    - This function assumes that the `block` dictionary contains keys such as "ext" for the image type,
      "image" for the image data, "bbox" for the bounding box, "width", and "height".
    - The function uses `ImageTypeEnum` to validate and standardize the image type.
    - Metadata is structured in a hierarchical format, including the page number, block number, and nearby objects.
    - The `pymupdf_exception_handler` decorator is used to handle any exceptions related to PyMuPDF operations.

    Examples
    --------
    >>> block = {
    >>>     "ext": "png",
    >>>     "image": b"...",  # binary image data
    >>>     "bbox": (100, 200, 300, 400),
    >>>     "width": 200,
    >>>     "height": 100,
    >>>     "number": 1
    >>> }
    >>> page_idx = 2
    >>> page_count = 10
    >>> source_metadata = {"source": "some_source"}
    >>> base_unified_metadata = {"metadata_key": "metadata_value"}
    >>> page_nearby_blocks = ["text_block_1", "text_block_2"]
    >>> result = _extract_image_from_imageblock(
    >>>     block, page_idx, page_count, source_metadata, base_unified_metadata, page_nearby_blocks
    >>> )
    >>> print(result)
    [ContentTypeEnum.IMAGE, {...}, 'a1b2c3d4-5678-90ab-cdef-1234567890ab']
    """
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
    content: str
    image: str
    bbox: Tuple[int, int, int, int]


@dataclass
class ImageChart:
    content: str
    image: str
    bbox: Tuple[int, int, int, int]


def _extract_tables_using_pymupdf(page: fitz.Page) -> List[DataFrameTable]:
    """
    Extract embedded tables from a PDF page using PyMuPDF.

    This function identifies tables on a PDF page by detecting vector lines (both horizontal and vertical)
    and converts these tables into pandas DataFrames. The function then cleans up the DataFrame columns by
    removing any newlines, tabs, or extra spaces in the column names. The resulting DataFrames, along with
    their corresponding bounding boxes, are returned as a list of DataFrameTable objects.

    Parameters
    ----------
    page : fitz.Page
        A page object from a PyMuPDF `Document`, representing the PDF page from which tables are to be extracted.

    Returns
    -------
    List[DataFrameTable]
        A list of DataFrameTable objects, each containing a pandas DataFrame of the extracted table and its
        corresponding bounding box on the page.

    Notes
    -----
    - This function relies on the `find_tables` method of PyMuPDF, which uses the strategy of detecting
      horizontal and vertical lines to identify tables.
    - The `horizontal_strategy="text"` option in `find_tables` is known to perform poorly and is therefore
      not used in this function.
    - The extracted DataFrame column names are sanitized by replacing any whitespace characters
      (e.g., newlines, tabs) with a single space, making them more suitable for further processing or
      conversion to other formats such as Markdown.

    Examples
    --------
    >>> import fitz
    >>> from some_module import _extract_tables_using_pymupdf
    >>> doc = fitz.open("sample.pdf")
    >>> page = doc.load_page(0)
    >>> tables = _extract_tables_using_pymupdf(page)
    >>> for table in tables:
    >>>     print(table.dataframe)
    >>>     print(table.bbox)
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


def run_image_inference(client, model_name, image_data):
    """
    Run inference on the given model using the provided client and cropped image.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        The gRPC client to use for inference.
    model_name : str
        The name of the model to infer with.
    image_data : np.ndarray
        The cropped image data to be inferred. It should be a numpy array with shape suitable
        for the model input.

    Returns
    -------
    Union[str, None]
        The result of the inference as a string, or None if the inference fails.

    Notes
    -----
    This function assumes that the model outputs a single string result. It logs an error
    if the inference fails.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> image_data = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    >>> result = run_image_inference(client, "my_model", image_data)
    >>> print(result)
    """
    inputs = [grpcclient.InferInput("input", image_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_data.astype(np.float32))

    outputs = [grpcclient.InferRequestedOutput("output")]

    try:
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        return " ".join([output[0].decode("utf-8") for output in result.as_numpy("output")])
    except Exception as e:
        logger.error(f"Inference failed for model {model_name}: {str(e)}")
        return None


def join_cached_and_deplot_output(cached_text, deplot_text):
    """
    Process the inference results from cached and deplot models.

    Parameters
    ----------
    cached_text : str
        The result from the cached model inference, expected to be a JSON string or plain text.
    deplot_text : str
        The result from the deplot model inference, expected to be plain text.

    Returns
    -------
    str
        The concatenated and processed chart content as a string.

    Notes
    -----
    This function attempts to parse the `cached_text` as JSON to extract specific fields.
    If parsing fails, it falls back to using the raw `cached_text`. The `deplot_text` is then
    appended to this content.

    Examples
    --------
    >>> cached_text = '{"chart_title": "Sales Over Time"}'
    >>> deplot_text = "This chart shows the sales over time."
    >>> result = join_cached_and_deplot_output(cached_text, deplot_text)
    >>> print(result)
    """
    chart_content = ""

    if (cached_text is not None) and (deplot_text is not None):
        try:
            cached_text_dict = json.loads(cached_text)
            chart_content += cached_text_dict.get("chart_title", "")
        except json.JSONDecodeError:
            chart_content += cached_text

        chart_content += deplot_text

    return chart_content


# Centralized client creation for handling retries and backoff
def create_inference_client(endpoint_url: str):
    """
    Create an inference client for communicating with a Triton server.

    Parameters
    ----------
    endpoint_url : str
        The URL of the Triton inference server.

    Returns
    -------
    grpcclient.InferenceServerClient
        A gRPC client for making inference requests to the Triton server.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> type(client)
    <class 'grpcclient.InferenceServerClient'>
    """
    return grpcclient.InferenceServerClient(url=endpoint_url)


# Process a batch of pages into images
def process_pages_to_images(pages: List[fitz.Page]) -> List[np.ndarray]:
    """
    Convert a list of document pages into image arrays.

    Parameters
    ----------
    pages : List[fitz.Page]
        A list of document pages to be converted into images.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays, where each array represents an image corresponding to a page.

    Notes
    -----
    The images are generated at a resolution of 300 DPI and resized to a maximum size of 1536x1536 pixels.

    Examples
    --------
    >>> pages = [fitz.Page(), fitz.Page()]
    >>> images = process_pages_to_images(pages)
    >>> type(images[0])
    <class 'numpy.ndarray'>
    """
    images = []
    for page in pages:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        img_bytes = pixmap.tobytes("png")
        with io.BytesIO(img_bytes) as f:
            img = Image.open(f)
            img.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            img_arr = np.array(img)
        images.append(img_arr)
    return images


# Prepare the images for model inference
def prepare_images_for_inference(images: List[np.ndarray]) -> np.ndarray:
    """
    Prepare a list of images for model inference by resizing and reordering axes.

    Parameters
    ----------
    images : List[np.ndarray]
        A list of image arrays to be prepared for inference.

    Returns
    -------
    np.ndarray
        A numpy array suitable for model input, with the shape reordered to match the expected input format.

    Notes
    -----
    The images are resized to 1024x1024 pixels and the axes are reordered to match the expected input shape for
    the model (batch, channels, height, width).

    Examples
    --------
    >>> images = [np.random.rand(1536, 1536, 3) for _ in range(2)]
    >>> input_array = prepare_images_for_inference(images)
    >>> input_array.shape
    (2, 3, 1024, 1024)
    """

    resized_images = [yolox_utils.resize_image(image, (1024, 1024)) for image in images]
    return np.einsum("bijk->bkij", resized_images).astype(np.float32)


# Perform inference and return predictions
def perform_model_inference(client, model_name: str, input_array: np.ndarray):
    """
    Perform inference using the provided model and input data.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        The gRPC client to use for inference.
    model_name : str
        The name of the model to use for inference.
    input_array : np.ndarray
        The input data to feed into the model, formatted as a numpy array.

    Returns
    -------
    np.ndarray
        The output of the model as a numpy array.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> input_array = np.random.rand(2, 3, 1024, 1024).astype(np.float32)
    >>> output = perform_model_inference(client, "my_model", input_array)
    >>> output.shape
    (2, 1000)
    """
    input_tensors = [grpcclient.InferInput("input", input_array.shape, datatype="FP32")]
    input_tensors[0].set_data_from_numpy(input_array)

    outputs = [grpcclient.InferRequestedOutput("output")]
    query_response = client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)

    return query_response.as_numpy("output")


# Process inference results and expand bounding boxes
def process_inference_results(
    output_array: np.ndarray,
    original_image_shapes: List[Tuple[int, int]],
    num_classes: int,
    conf_thresh: float,
    iou_thresh: float,
    min_score: float,
):
    """
    Process the model output to generate detection results and expand bounding boxes.

    Parameters
    ----------
    output_array : np.ndarray
        The raw output from the model inference.
    original_image_shapes : List[Tuple[int, int]]
        The shapes of the original images before resizing, used for scaling bounding boxes.
    num_classes : int
        The number of classes the model can detect.
    conf_thresh : float
        The confidence threshold for detecting objects.
    iou_thresh : float
        The Intersection Over Union (IoU) threshold for non-maximum suppression.
    min_score : float
        The minimum score for keeping a detection.

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing processed detection results including expanded bounding boxes.

    Notes
    -----
    This function applies non-maximum suppression to the model's output and scales the bounding boxes back to the
    original image size.

    Examples
    --------
    >>> output_array = np.random.rand(2, 100, 85)
    >>> original_image_shapes = [(1536, 1536), (1536, 1536)]
    >>> results = process_inference_results(output_array, original_image_shapes, 80, 0.5, 0.5, 0.1)
    >>> len(results)
    2
    """
    pred = yolox_utils.postprocess_model_prediction(
        output_array, num_classes, conf_thresh, iou_thresh, class_agnostic=True
    )
    results = yolox_utils.postprocess_results(pred, original_image_shapes, min_score=min_score)
    return [yolox_utils.expand_chart_bboxes(annotation_dict) for annotation_dict in results]


# Handle individual table/chart extraction and model inference
def handle_table_chart_extraction(
    annotation_dict, original_image, page_idx, paddle_client, deplot_client, cached_client, tables_and_charts
):
    """
    Handle the extraction of tables and charts from the inference results and run additional model inference.

    Parameters
    ----------
    annotation_dict : dict
        A dictionary containing detected objects and their bounding boxes.
    original_image : np.ndarray
        The original image from which objects were detected.
    page_idx : int
        The index of the current page being processed.
    paddle_client : grpcclient.InferenceServerClient
        The gRPC client for the paddle model used to process tables.
    deplot_client : grpcclient.InferenceServerClient
        The gRPC client for the deplot model used to process charts.
    cached_client : grpcclient.InferenceServerClient
        The gRPC client for the cached model used to process charts.
    tables_and_charts : List[Tuple[int, ImageTable]]
        A list to which extracted tables and charts will be appended.

    Notes
    -----
    This function iterates over detected objects, crops the original image to the bounding boxes,
    and runs additional inference on the cropped images to extract detailed information about tables
    and charts.

    Examples
    --------
    >>> annotation_dict = {"table": [], "chart": []}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> tables_and_charts = []
    >>> handle_table_chart_extraction(annotation_dict, original_image, 0, paddle_client, deplot_client, cached_client,
    tables_and_charts)
    """
    width, height, *_ = original_image.shape
    for label in ["table", "chart"]:
        if not annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = bbox * np.array([height, width, height, width])
            cropped = original_image[floor(w1) : ceil(w2), floor(h1) : ceil(h2)]  # noqa: E203
            img = Image.fromarray(cropped.astype(np.uint8))
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                base64_img = bytetools.base64frombytes(buffer.getvalue())

            if label == "table":
                table_content = run_image_inference(paddle_client, "paddle", np.expand_dims(cropped, axis=0))
                table_data = ImageTable(table_content, base64_img, (w1, h1, w2, h2))
                tables_and_charts.append((page_idx, table_data))
            elif label == "chart":
                deplot_result = run_image_inference(deplot_client, "deplot", np.expand_dims(cropped, axis=0))
                cached_result = run_image_inference(cached_client, "cached", np.expand_dims(cropped, axis=0))
                chart_content = join_cached_and_deplot_output(cached_result, deplot_result)
                chart_data = ImageChart(chart_content, base64_img, (w1, h1, w2, h2))
                tables_and_charts.append((page_idx, chart_data))


# Main extraction function that encapsulates the entire process
def extract_tables_and_charts_using_image_ensemble(
    pages: List[fitz.Page],
    yolox_nim_endpoint_url: str,
    model_name: str,
    cached_nim_endpoint_url: str,
    deplot_nim_endpoint_url: str,
    paddle_nim_endpoint_url: str,
    max_batch_size: int = 8,
    num_classes: int = 3,
    conf_thresh: float = 0.48,
    iou_thresh: float = 0.5,
    min_score: float = 0.1,
) -> List[Tuple[int, ImageTable]]:
    """
    Extract tables and charts from a series of document pages using an ensemble of image-based models.

    This function processes a list of document pages to detect and extract tables and charts.
    It uses a sequence of models hosted on different inference servers to achieve this.

    Parameters
    ----------
    pages : List[fitz.Page]
        A list of document pages to process.
    yolox_nim_endpoint_url : str
        The URL of the Triton inference server endpoint for the primary model.
    model_name : str
        The name of the model to use on the Triton server.
    max_batch_size : int, optional
        The maximum number of pages to process in a single batch (default is 16).
    num_classes : int, optional
        The number of classes the model is trained to detect (default is 3).
    conf_thresh : float, optional
        The confidence threshold for detection (default is 0.48).
    iou_thresh : float, optional
        The Intersection Over Union (IoU) threshold for non-maximum suppression (default is 0.5).
    min_score : float, optional
        The minimum score threshold for considering a detection valid (default is 0.1).

    Returns
    -------
    List[Tuple[int, ImageTable]]
        A list of tuples, each containing the page index and an `ImageTable` or `ImageChart` object
        representing the detected table or chart along with its associated metadata.

    Notes
    -----
    This function centralizes the management of inference clients, handles batch processing
    of pages, and manages the inference and post-processing of results from multiple models.
    It ensures that the results are properly associated with their corresponding pages and
    regions within those pages.

    Examples
    --------
    >>> pages = [fitz.Page(), fitz.Page()]  # List of pages from a document
    >>> tables_and_charts = extract_tables_and_charts_using_image_ensemble(
    ...     pages,
    ...     yolox_nim_endpoint_url="http://localhost:8000",
    ...     model_name="model",
    ...     max_batch_size=8,
    ...     num_classes=3,
    ...     conf_thresh=0.5,
    ...     iou_thresh=0.5,
    ...     min_score=0.2
    ... )
    >>> for page_idx, image_obj in tables_and_charts:
    ...     print(f"Page: {page_idx}, Object: {image_obj}")
    """
    tables_and_charts = []

    triton_client = paddle_client = deplot_client = cached_client = None
    try:
        cached_client = create_inference_client(cached_nim_endpoint_url)
        deplot_client = create_inference_client(deplot_nim_endpoint_url)
        paddle_client = create_inference_client(paddle_nim_endpoint_url)
        triton_client = create_inference_client(yolox_nim_endpoint_url)

        batches = []
        i = 0
        while i < len(pages):
            batch_size = min(2 ** int(log(len(pages) - i, 2)), max_batch_size)
            batches.append(pages[i : i + batch_size])  # noqa: E203
            i += batch_size

        page_idx = 0
        for batch in batches:
            original_images = process_pages_to_images(batch)
            original_image_shapes = [image.shape for image in original_images]
            input_array = prepare_images_for_inference(original_images)

            output_array = perform_model_inference(triton_client, model_name, input_array)
            results = process_inference_results(
                output_array, original_image_shapes, num_classes, conf_thresh, iou_thresh, min_score
            )

            # assert len(results) == len(batch)

            for annotation_dict, original_image in zip(results, original_images):
                handle_table_chart_extraction(
                    annotation_dict,
                    original_image,
                    page_idx,
                    paddle_client,
                    deplot_client,
                    cached_client,
                    tables_and_charts,
                )

                page_idx += 1
    finally:
        if paddle_client:
            paddle_client.close()
        if deplot_client:
            deplot_client.close()
        if cached_client:
            cached_client.close()
        if triton_client:
            triton_client.close()

    return tables_and_charts


# TODO(Devin): Disambiguate tables and charts, create two distinct processing methods
@pymupdf_exception_handler(descriptor="pymupdf")
def _construct_table_or_chart_metadata(
    table: Union[DataFrameTable, ImageTable, ImageChart],
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    """
    +--------------------------------+--------------------------+------------+---+
    | Table/Chart Metadata           |                          | Extracted  | Y |
    | (tables within documents)      |                          |            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table format                   | Structured (dataframe /  | Extracted  |   |
    |                                | lists of rows and        |            |   |
    |                                | columns), or serialized  |            |   |
    |                                | as markdown, html,       |            |   |
    |                                | latex, simple (cells     |            |   |
    |                                | separated just as spaces)|            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table content                  | Extracted text content   |            |   |
    |                                |                          |            |   |
    |                                | Important: Tables should |            |   |
    |                                | not be chunked           |            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table location                 | Bounding box of the table|            |   |
    +--------------------------------+--------------------------+------------+---+
    | Caption                        | Detected captions for    |            |   |
    |                                | the table/chart          |            |   |
    +--------------------------------+--------------------------+------------+---+
    | uploaded_image_uri             | Mirrors                  |            |   |
    |                                | source_metadata.         |            |   |
    |                                | source_location          |            |   |
    +--------------------------------+--------------------------+------------+---+
    """

    if isinstance(table, DataFrameTable):
        content = table.df.to_markdown(index=False)
        structured_content_text = content
        table_format = TableFormatEnum.MARKDOWN
        subtype = ContentSubtypeEnum.TABLE
        description = StdContentDescEnum.PDF_TABLE

    elif isinstance(table, ImageTable):
        content = table.image
        structured_content_text = table.content
        table_format = TableFormatEnum.IMAGE
        subtype = ContentSubtypeEnum.TABLE
        description = StdContentDescEnum.PDF_TABLE

    elif isinstance(table, ImageChart):
        content = table.image
        structured_content_text = table.content
        table_format = TableFormatEnum.IMAGE
        subtype = ContentSubtypeEnum.CHART
        description = StdContentDescEnum.PDF_CHART

    else:
        raise ValueError("Unknown table/chart type.")

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
        "table_content": structured_content_text,
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
