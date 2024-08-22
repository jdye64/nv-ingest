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
import logging
from math import ceil
from math import floor
from math import log
from typing import List
from typing import Tuple

import numpy as np
import pypdfium2 as libpdfium
from PIL import Image

from nv_ingest.extraction_workflows.pdf import yolox_utils
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.util.converters import bytetools
from nv_ingest.util.image_processing.table_and_chart import join_cached_and_deplot_output
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import ImageChart
from nv_ingest.util.pdf.metadata_aggregators import ImageTable
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy
from nv_ingest.util.triton.helpers import call_image_inference_model
from nv_ingest.util.triton.helpers import create_inference_client
from nv_ingest.util.triton.helpers import perform_model_inference

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


def extract_tables_and_charts_using_image_ensemble(
    pages: List[libpdfium.PdfPage],
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
    pages : List[libpdfium.PdfPage]
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
    >>> pages = [libpdfium.PdfPage(), libpdfium.PdfPage()]  # List of pages from a document
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
            original_images, _ = pdfium_pages_to_numpy(batch, scale_tuple=(1536, 1536))

            original_image_shapes = [image.shape for image in original_images]
            input_array = prepare_images_for_inference(original_images)

            output_array = perform_model_inference(triton_client, model_name, input_array)
            results = process_inference_results(
                output_array, original_image_shapes, num_classes, conf_thresh, iou_thresh, min_score
            )

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
    except Exception as e:
        logger.error(f"Error during table/chart extraction: {str(e)}")
        raise
    finally:
        if paddle_client:
            paddle_client.close()
        if deplot_client:
            deplot_client.close()
        if cached_client:
            cached_client.close()
        if triton_client:
            triton_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts.")

    return tables_and_charts


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
                table_content = call_image_inference_model(paddle_client, "paddle", np.expand_dims(cropped, axis=0))
                table_data = ImageTable(table_content, base64_img, (w1, h1, w2, h2))
                tables_and_charts.append((page_idx, table_data))
            elif label == "chart":
                deplot_result = call_image_inference_model(deplot_client, "deplot", np.expand_dims(cropped, axis=0))
                cached_result = call_image_inference_model(cached_client, "cached", np.expand_dims(cropped, axis=0))
                chart_content = join_cached_and_deplot_output(cached_result, deplot_result)
                chart_data = ImageChart(chart_content, base64_img, (w1, h1, w2, h2))
                tables_and_charts.append((page_idx, chart_data))


# Define a helper function to use unstructured-io to extract text from a base64
# encoded bytestram PDF
def pdfium(pdf_stream, extract_text: bool, extract_images: bool, extract_tables: bool, **kwargs):
    """
    Helper function to use pdfium to extract text from a bytestream PDF.

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

    logger.debug("Extracting PDF with pdfium backend.")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    kwargs.get("identify_nearby_objects", True)

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

    extracted_data = []
    doc = libpdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)

    source_metadata = {
        "source_name": pdf_metadata.filename,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": pdf_metadata.source_type,
        "collection_id": collection_id,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    logger.debug(f"Extracting text from PDF with {pdf_metadata.page_count} pages.")
    logger.debug(f"Extract text: {extract_text}")
    logger.debug(f"extract images: {extract_images}")
    logger.debug(f"extract tables: {extract_tables}")

    text_depth = TextTypeEnum.DOCUMENT  # TODO(Devin): Add page level text extraction
    for page_idx in range(pdf_metadata.page_count):
        accumulated_text = []
        page = doc.get_page(page_idx)

        # https://pypdfium2.readthedocs.io/en/stable/python_api.html#module-pypdfium2._helpers.textpage
        if extract_text:
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            accumulated_text.append(page_text)
            text_extraction = construct_text_metadata(
                accumulated_text,
                pdf_metadata.keywords,
                page_idx,
                -1,
                -1,
                -1,
                pdf_metadata.page_count,
                text_depth,
                source_metadata,
                base_unified_metadata,
            )

            extracted_data.append(text_extraction)

        # Image extraction
        if extract_images:
            for obj in page.get_objects():
                obj_type = PDFIUM_PAGEOBJ_MAPPING.get(obj.type, "UNKNOWN")
                if obj_type == "IMAGE":
                    try:
                        # Attempt to retrieve the image bitmap
                        image_numpy: np.ndarray = pdfium_try_get_bitmap_as_numpy(obj)
                        image_base64: str = numpy_to_base64(image_numpy)
                        image_bbox = obj.get_pos()
                        image_size = obj.get_size()
                        image_data = Base64Image(image_base64, image_bbox, image_size[0], image_size[1])

                        extracted_image_data = construct_image_metadata(
                            image_data,
                            page_idx,
                            pdf_metadata.page_count,
                            source_metadata,
                            base_unified_metadata,
                        )

                        extracted_data.append(extracted_image_data)
                    except Exception as e:
                        logger.error(f"Error extracting image: {e}")
                        pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

        # Table and chart collection
        if extract_tables:
            pages.append(page)

    if extract_tables:
        for page_idx, table_and_charts in extract_tables_and_charts_using_image_ensemble(
            pages,
            table_detection_endpoint_url,
            table_detection_model_name,
            cached_nim_endpoint_url,
            deplot_nim_endpoint_url,
            paddle_nim_endpoint_url,
        ):
            extracted_data.append(
                construct_table_and_chart_metadata(
                    table_and_charts, page_idx, pdf_metadata.page_count, source_metadata, base_unified_metadata
                )
            )

    logger.debug(f"Extracted {len(extracted_data)} items from PDF.")

    return extracted_data
