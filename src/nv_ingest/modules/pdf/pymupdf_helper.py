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
from datetime import datetime

from nv_ingest.util.converters import bytetools
from nv_ingest.schemas.metadata import ContentTypeEnum
from nv_ingest.schemas.metadata import SourceTypeEnum
from nv_ingest.schemas.metadata import TextTypeEnum
from nv_ingest.schemas.metadata import ImageTypeEnum
from nv_ingest.schemas.metadata import  AccessLevelEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.schemas.metadata import StdContentDescEnum
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.converters import datetools
from nv_ingest.util.detectors.language import detect_language
from nv_ingest.util.exception_handlers.pdf import pymupdf_exception_handler

import fitz

logger = logging.getLogger(__name__)


# Define a helper function to use unstructured-io to extract text from a base64 encoded bytestram PDF
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

    logger.debug(f"Extracting PDF with PyMuPDF backend.")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]
    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")

    base_unified_metadata = \
        row_data[metadata_col] if metadata_col in row_data.index else {}
    
    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)  
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get(
        "access_level", AccessLevelEnum.LEVEL_1)

    # each row is a partition level document
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:

        extracted_data = []
        page_count = doc.page_count
        filename = doc.name

        # last_modified
        last_modified = doc.metadata.get("modDate", None)
        if last_modified in (None, "",):
            last_modified = datetools.remove_tz(datetime.now()).isoformat()
        else:
            last_modified = datetools.datetimefrompdfmeta(last_modified)

        # date_created
        date_created = doc.metadata.get("creationDate", None)
        if date_created in (None, "",):
            date_created =  datetools.remove_tz(datetime.now()).isoformat()
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

            page = doc[page_idx]      
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]  # the list of block dictionaries

            for block in blocks:

                # Extract text (a) - block/line/span
                if (extract_text) and (block["type"] == 0):

                    block_idx = block["number"]
                    for line_idx, line in enumerate(block["lines"]): # lines a list                                        
                        for span_idx, span in enumerate(line["spans"]): # spans is a list                    
                            accumulated_text.append(span["text"])

                            if (text_depth == TextTypeEnum.SPAN):

                                text_extraction = _construct_text_metadata(
                                    accumulated_text, 
                                    keywords, 
                                    page_idx, 
                                    block_idx, 
                                    line_idx, 
                                    span_idx, 
                                    page_count, 
                                    text_depth, 
                                    source_metadata, 
                                    base_unified_metadata)   
                                
                                if (len(text_extraction) > 0):
                                    extracted_data.append(text_extraction)

                                accumulated_text = []                  

                        if text_depth == TextTypeEnum.LINE:

                            text_extraction = _construct_text_metadata(
                                accumulated_text, 
                                keywords, 
                                page_idx, 
                                block_idx, 
                                line_idx, 
                                -1, 
                                page_count, 
                                text_depth, 
                                source_metadata, 
                                base_unified_metadata)
                            
                            if len(text_extraction) > 0:
                                extracted_data.append(text_extraction)

                            accumulated_text = []

                    if text_depth == TextTypeEnum.BLOCK:

                        text_extraction = _construct_text_metadata(
                            accumulated_text, 
                            keywords, 
                            page_idx, 
                            block_idx, 
                            -1, 
                            -1, 
                            page_count, 
                            text_depth, 
                            source_metadata, 
                            base_unified_metadata)
                        
                        if len(text_extraction) > 0:
                            extracted_data.append(text_extraction)                

                        accumulated_text = []                         

                # Extract images
                if (extract_images) and (block["type"] == 1):

                    image_extraction = _extract_image(
                        block, 
                        page_idx, 
                        page_count, 
                        source_metadata, 
                        base_unified_metadata)
                    
                    extracted_data.append(image_extraction)

            # Extract text - page (b) 
            if (extract_text) and (text_depth == TextTypeEnum.PAGE):

                text_extraction = _construct_text_metadata(
                    accumulated_text, 
                    keywords, 
                    page_idx, 
                    -1, 
                    -1, 
                    -1, 
                    page_count, 
                    text_depth, 
                    source_metadata, 
                    base_unified_metadata)
                
                if len(text_extraction) > 0:
                    extracted_data.append(text_extraction)            

                accumulated_text = []

            # extract page tables
            if (extract_tables):
                pass

        # Extract text - document (c) 
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
                base_unified_metadata)
            
            if len(text_extraction) > 0:
                extracted_data.append(text_extraction)            

            accumulated_text = []

    return extracted_data


@pymupdf_exception_handler(descriptor="pymupdf")
def _construct_text_metadata(
        accumulated_text, keywords, page_idx, block_idx, 
        line_idx, span_idx, page_count, text_depth, source_metadata, 
        base_unified_metadata):

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
            }
        }

    language = detect_language(extracted_text)

    text_metadata = {
        "text_type": text_depth,
        "summary": "",
        "keywords": keywords,
        "language": language,
        }
    
    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update({
        "content": extracted_text,
        "source_metadata": source_metadata,
        "content_metadata": content_metadata,
        "text_metadata": text_metadata,
        })

    validated_unified_metadata = validate_schema(
        ext_unified_metadata, PDFExtractorSchema)    

    return [ContentTypeEnum.TEXT, validated_unified_metadata.dict()]


@pymupdf_exception_handler(descriptor="pymupdf")
def _extract_image(block, page_idx, page_count, source_metadata, base_unified_metadata):

    image_type = block["ext"]
    if ImageTypeEnum.has_value(image_type):
        image_type = ImageTypeEnum[image_type.upper()]

    base64_img = bytetools.base64frombytes(block["image"])

    bbox = block["bbox"]
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
            }
        }      

    image_metadata = {
        "image_type": image_type,
        "structured_image_type": ImageTypeEnum.image_type_1,
        "caption": "",
        "text": "",
        "image_location": bbox,
        }              

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update({
        "content": base64_img,
        "source_metadata": source_metadata,
        "content_metadata": content_metadata,
        "image_metadata": image_metadata,
        })
    
    validated_unified_metadata = validate_schema(
        unified_metadata, PDFExtractorSchema)   

    return [ContentTypeEnum.IMAGE, validated_unified_metadata.dict()]
