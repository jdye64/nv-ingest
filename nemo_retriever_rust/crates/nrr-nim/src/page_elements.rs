// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Page-elements NIM (yolox-page-elements-v3) client.
//!
//! Wraps [`crate::ocr::invoke_image_endpoint`] with a stable label so
//! Prometheus metrics keyed on the endpoint name remain consistent with
//! the Python implementation.

use bytes::Bytes;
use serde_json::Value as JsonValue;

use crate::client::{NimClient, NimError};

pub async fn detect(
    client: &NimClient,
    endpoint: &str,
    image_bytes: &Bytes,
) -> Result<JsonValue, NimError> {
    crate::ocr::invoke_image_endpoint(client, endpoint, image_bytes, "page_elements_invoke_url")
        .await
}
