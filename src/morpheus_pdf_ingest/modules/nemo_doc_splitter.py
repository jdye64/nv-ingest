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
import json
from typing import List, Literal, Any

import mrc
import pandas as pd
from more_itertools import windowed
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from pydantic import ValidationError

from morpheus_pdf_ingest.schemas.nemo_doc_splitter_schema import DocumentSplitterSchema

logger = logging.getLogger(__name__)


def _build_split_documents(row, text_splits: List[str], sentence_window_size: int) -> List[dict[str, Any]]:
    """Build documents from text splits with window text."""
    documents: List[dict] = []

    window_size = sentence_window_size
    for i, text in enumerate(text_splits):
        if text is None or not text.strip():
            continue

        metadata = row.meta if isinstance(row.meta, dict) else {}
        metadata["source_id"] = row.id
        if window_size > 0:
            window_text = "".join(
                text_splits[
                max(0, i - window_size): min(i + 1 + window_size, len(text_splits))
                ]
            )
            metadata["window"] = window_text
            metadata["original_text"] = text
        documents.append({"content": text, "meta": metadata})

    return documents


def _split_into_units(text: str, split_by: Literal["word", "sentence", "passage"]) -> List[str]:
    if split_by == "passage":
        split_at = "\n\n"
    elif split_by == "sentence":
        split_at = "."  # why not ?,!, etc..?
    elif split_by == "word":
        split_at = " "
    else:
        raise NotImplementedError(
            "DocumentSplitter only supports 'passage', 'sentence'"
            " or 'word' split_by options."
        )
    units = text.split(split_at)
    # Add the delimiter back to all units except the last one
    for i in range(len(units) - 1):
        units[i] += split_at

    return units


def _concatenate_units(units: List[str], split_length: int, split_overlap: int, max_character_length: int) -> List[
    str]:
    text_splits = []
    segments = windowed(units, n=split_length, step=split_length - split_overlap)
    for seg in segments:
        current_units = [unit for unit in seg if unit is not None]
        txt = "".join(current_units)
        if (max_character_length and len(txt) > max_character_length):
            text_splits.extend(_split_long_text(txt, max_character_length))
        elif len(txt) > 0:
            text_splits.append(txt)

    return text_splits


def _split_long_text(text: str, max_character_length: int) -> List[str]:
    """
    Splits a long text into smaller segments that
    do not exceed max_character_length.
    """
    split_texts = []
    while text:
        # Take the maximum possible substring without exceeding max_character_length
        segment = text[: max_character_length]
        split_texts.append(segment)
        text = text[max_character_length:]  # noqa: E203

    return split_texts


def process_content(row, validated_config):
    units = _split_into_units(row['content'], validated_config.split_by)
    text_splits = _concatenate_units(units, validated_config.split_length, validated_config.split_overlap,
                                     max_character_length=validated_config.max_character_length)
    split_docs = _build_split_documents(row, text_splits,
                                        sentence_window_size=validated_config.sentence_window_size)

    return split_docs


NemoDocSplitterLoaderFactory = ModuleLoaderFactory("nemo_document_splitter", "morpheus_pdf_ingest",
                                                   DocumentSplitterSchema)


@register_module("nemo_document_splitter", "morpheus_pdf_ingest")
def _nemo_document_splitter(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """

    module_config = builder.get_current_module_config()
    try:
        validated_config = DocumentSplitterSchema(**module_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid Nemo Document Splitter configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    def split_and_forward(message: ControlMessage):
        # Assume that df is going to have a 'content' column

        split_docs = []

        # Validate that all 'content' values are not None
        df = message.payload().df.to_pandas()
        if df['content'].isnull().any():
            raise ValueError(
                "DocumentSplitter only works with text documents but one or more 'content' values are None.")

        split_docs = []
        for _, row in df.iterrows():
            units = _split_into_units(row['content'], validated_config.split_by)
            text_splits = _concatenate_units(units, validated_config.split_length, validated_config.split_overlap,
                                             max_character_length=validated_config.max_character_length)
            split_docs.extend(_build_split_documents(row, text_splits,
                                                     sentence_window_size=validated_config.sentence_window_size))
        split_docs_df = pd.DataFrame(split_docs)
        message_meta = MessageMeta(df=split_docs_df)
        message.payload(message_meta)

        return message

    split_node = builder.make_node("split_and_forward", ops.map(split_and_forward))

    # Register the input and output of the module
    builder.register_module_input("input", split_node)
    builder.register_module_output("output", split_node)
