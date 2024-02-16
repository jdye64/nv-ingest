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
    """    

    logger.info(f"Extracting PDF with PyMuPDF backend.")

    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text_list = []
    # Extract text from each page
    for page in doc:
        text_list.append(page.get_text())

    doc.close()  # Close the document

    text = "".join(text_list).replace('+', ' ')

    return text
