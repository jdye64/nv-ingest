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

from morpheus_pdf_ingest.util.converters import bytetools
from morpheus_pdf_ingest.schemas.metadata import ExtractedDocumentType

import numpy as np

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
    metadata_column = kwargs.get("metadata_column", "metadata")
    metadata = row_data[metadata_column] if metadata_column in row_data.index else {}
    idx = row_data.name

    # each row is a partition level document
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:

        extracted_data = []

        for page_idx in range(len(doc)):

            page = doc[page_idx]
            
            # extract page text
            if (extract_text):
                page_text = page.get_text().replace('+', ' ')

                # append only text was extracted
                if len(page_text) > 1:
                    
                    page_txt_metadata = metadata.copy()
                    page_txt_metadata.update({
                        "page_number": page_idx,
                        "document_type": ExtractedDocumentType.text
                        })                

                    page_txt_metadata = {
                        "content": page_text,
                        "metadata": page_txt_metadata,
                        }
                    
                    page_text_payload = [
                        ExtractedDocumentType.text, 
                        page_txt_metadata
                        ]

                    extracted_data.append(page_text_payload)

            # extract page unstructured images
            if (extract_images):

                img_list = page.get_images()

                if img_list:

                    #loop over images on page
                    for img_idx, img in enumerate(img_list, start=1):
                        
                        # get the XREF of the image
                        xref = img[0] 
                        # extract image
                        pix = doc.extract_image(xref)
                        # convert image bytes to hex to work with cuDF
                        unstr_img_hex = bytetools.hexfrombytes(pix["image"])
                        # append unstructured image extraction
                        page_unstr_img_metadata = metadata.copy()
                        page_unstr_img_metadata.update({
                            "page_number": page_idx,
                            "document_type": ExtractedDocumentType.unstructured_image
                            })                

                        page_unstr_img_metadata = {
                            "content": unstr_img_hex,
                            "metadata": page_unstr_img_metadata,
                            }
                        
                        page_unstr_img_payload = [
                            ExtractedDocumentType.unstructured_image, 
                            page_unstr_img_metadata,
                            ]

                        extracted_data.append(page_unstr_img_payload)

            # extract page structured images
            if (extract_tables):
                pass

    return extracted_data
