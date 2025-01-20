# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest.util.converters.formats import ingest_json_results_to_blob


def test_json_results_to_blob_text_failure():
    # there must be a "data" element in the json otherwise empty is returned
    blob_response = ingest_json_results_to_blob("something")
    assert blob_response == ""
