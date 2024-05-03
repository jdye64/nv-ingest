from io import BytesIO

import pandas as pd
import pytest

from nv_ingest.extraction_workflows.pdf.pymupdf_helper import pymupdf
from nv_ingest.schemas.metadata_schema import TextTypeEnum


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
        }
    )


@pytest.fixture
def pdf_stream():
    with open("data/test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


def test_pymupdf_basic(pdf_stream, document_df):
    extracted_data = pymupdf(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"


@pytest.mark.parametrize(
    "text_depth",
    ["span", TextTypeEnum.SPAN, "line", TextTypeEnum.LINE, "block", TextTypeEnum.BLOCK],
)
def test_pymupdf_text_depth_line(pdf_stream, document_df, text_depth):
    extracted_data = pymupdf(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 3
    assert all(len(x) == 3 for x in extracted_data)
    assert all(x[0].value == "text" for x in extracted_data)
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"] == "Here is one line of text."
    assert extracted_data[1][1]["content"] == "Here is another line of text."
    assert extracted_data[2][1]["content"] == "Here is an image."
    assert all(x[1]["source_metadata"]["source_id"] == "source1" for x in extracted_data)


@pytest.mark.parametrize(
    "text_depth",
    ["page", TextTypeEnum.PAGE, "document", TextTypeEnum.DOCUMENT],
)
def test_pymupdf_text_depth_page(pdf_stream, document_df, text_depth):
    extracted_data = pymupdf(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"


def test_pymupdf_extract_image(pdf_stream, document_df):
    extracted_data = pymupdf(
        pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "image"
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
    assert extracted_data[1][0].value == "text"
    assert (
        extracted_data[1][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
