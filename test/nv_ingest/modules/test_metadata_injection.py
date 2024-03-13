# Disabled while we figure out how to get morpheus deps into the environment
# import cudf
# import pandas as pd
# import pytest
# from morpheus.messages import ControlMessage, MessageMeta
#
# from nv_ingest.modules.metadata_injector import (
#    _on_data as on_data,
# )  # Import your module here
# from nv_ingest.schemas.ingest_job import DocumentTypeEnum
# from nv_ingest.schemas.metadata import ContentTypeEnum, MetadataSchema
# from nv_ingest.util.converters.type_mappings import DOC_TO_CONTENT_MAP
#
# Mapping of DocumentTypeEnum values to expected ContentTypeEnum values
#
#
# @pytest.fixture
# def document_df():
#    """Fixture to create a DataFrame for testing."""
#    return pd.DataFrame(
#        {
#            "document_type": [],
#            "content": [],
#            "source_id": [],
#            "source_name": [],
#        }
#    )
#
#
# @pytest.mark.parametrize("doc_type, expected_content_type", DOC_TO_CONTENT_MAP.items())
# def test_on_data_injects_correct_metadata_and_validates_schema(
#    document_df, doc_type, expected_content_type
# ):
#    document_df = document_df.copy()
#    document_df["document_type"] = [doc_type.value]
#    document_df["content"] = ["Dummy content for testing"]
#    document_df["source_id"] = ["source1"]
#    document_df["source_name"] = ["Source One"]
#
#    message_meta = MessageMeta(df=cudf.from_pandas(document_df))
#    message = ControlMessage()
#    message.payload(message_meta)
#
#    updated_message = on_data(message)
#    with updated_message.payload().mutable_dataframe() as mdf:
#        updated_df = mdf.to_pandas()
#
#    for _, row in updated_df.iterrows():
#        metadata = row["metadata"]
#        validated_metadata = MetadataSchema(**metadata)
#        assert (
#            validated_metadata.content_metadata.type == expected_content_type.value
#        ), (
#            f"Document type {doc_type.value}"
#            f" should have content type {expected_content_type}"
#        )
#
#
# @pytest.mark.parametrize(
#    "doc_type",
#    [dt for dt in DocumentTypeEnum if DOC_TO_CONTENT_MAP[dt] != ContentTypeEnum.IMAGE],
# )
# def test_on_data_non_image_types_have_no_image_metadata(document_df, doc_type):
#    document_df["document_type"] = [doc_type.value]
#    document_df["content"] = ["Content irrelevant for this test"]
#    document_df["source_id"] = ["source1"]
#    document_df["source_name"] = ["Source One"]
#
#    message_meta = MessageMeta(df=cudf.from_pandas(document_df))
#    message = ControlMessage()
#    message.payload(message_meta)
#
#    updated_message = on_data(message)
#    with updated_message.payload().mutable_dataframe() as mdf:
#        updated_df = mdf.to_pandas()
#
#    for _, row in updated_df.iterrows():
#        assert (
#            row["metadata"]["image_metadata"] is None
#        ), f"image_metadata should be None for non-image content types, failed for {doc_type}"
#
#
# def test_metadata_schema_validation(document_df):
#    """Test that the injected metadata adheres to the expected schema."""
#    document_df["document_type"] = "pdf"
#
#    message_meta = MessageMeta(df=cudf.from_pandas(document_df))
#    message = ControlMessage()
#    message.payload(message_meta)
#
#    updated_message = on_data(message)
#    with updated_message.payload().mutable_dataframe() as mdf:
#        updated_df = mdf.to_pandas()
#
#    for _, row in updated_df.iterrows():
#        metadata = row["metadata"]
#        assert isinstance(metadata["content"], str), "Content should be a string."
#        assert isinstance(
#            metadata["content_metadata"], dict
#        ), "Content metadata should be a dictionary."
#        assert (
#            "type" in metadata["content_metadata"]
#        ), "Content metadata should include a type."
#        # Add more schema validation checks as necessary
#
#
# def test_handling_missing_required_fields(document_df):
#    """Test how missing required fields are handled."""
#    document_df.drop(
#        "document_type", axis=1, inplace=True
#    )  # Simulate missing 'document_type'
#    document_df["content"] = ["Dummy content for testing"]
#    document_df["source_id"] = ["source1"]
#    document_df["source_name"] = ["Source One"]
#
#    message_meta = MessageMeta(df=cudf.from_pandas(document_df))
#    message = ControlMessage()
#    message.payload(message_meta)
#
#    with pytest.raises(KeyError):
#        _ = on_data(message)
#
#
# def test_unsupported_document_type(document_df):
#    """Verify that a ValueError is raised with unsupported document types."""
#    document_df["document_type"] = "unsupported_type"
#    document_df["content"] = ["Dummy content for testing"]
#    document_df["source_id"] = ["source1"]
#    document_df["source_name"] = ["Source One"]
#
#    message_meta = MessageMeta(df=cudf.from_pandas(document_df))
#    message = ControlMessage()
#    message.payload(message_meta)
#
#    # Expect a ValueError due to the unsupported document type
#    with pytest.raises(ValueError):
#        updated_message = on_data(message)
#        with updated_message.payload().mutable_dataframe() as mdf:
#            _ = mdf.to_pandas()
