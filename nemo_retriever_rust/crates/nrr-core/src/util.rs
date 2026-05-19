// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Utility helpers shared by the rest of `nrr-core`.

use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// ISO-8601 UTC timestamp at microsecond precision (matches Python's
/// `datetime.now(timezone.utc).isoformat()` output).
pub fn utc_now_iso() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// Hex-encoded SHA-256 digest of the supplied bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Filename-extension based file classification, mirroring
/// `nemo_retriever.service.utils.file_type.FileClassifier` at a coarse level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileCategory {
    Pdf,
    Image,
    Audio,
    Video,
    Text,
    Html,
    Office,
    Unknown,
}

impl FileCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            FileCategory::Pdf => "pdf",
            FileCategory::Image => "image",
            FileCategory::Audio => "audio",
            FileCategory::Video => "video",
            FileCategory::Text => "text",
            FileCategory::Html => "html",
            FileCategory::Office => "office",
            FileCategory::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub filename: String,
    pub category: FileCategory,
    pub content_type: String,
}

/// Classify a file by its extension and (optionally) its declared content
/// type. The declared content type wins when present.
pub fn classify_file(filename: &str, content_type: Option<&str>) -> Classification {
    let lower = filename.to_ascii_lowercase();
    let ext = std::path::Path::new(&lower)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let category = match ext {
        "pdf" => FileCategory::Pdf,
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp" => FileCategory::Image,
        "mp3" | "wav" | "flac" | "ogg" | "m4a" => FileCategory::Audio,
        "mp4" | "mov" | "avi" | "mkv" | "webm" => FileCategory::Video,
        "txt" | "md" | "rst" | "csv" | "tsv" | "json" | "jsonl" => FileCategory::Text,
        "html" | "htm" => FileCategory::Html,
        "doc" | "docx" | "ppt" | "pptx" | "xls" | "xlsx" | "odt" => FileCategory::Office,
        _ => FileCategory::Unknown,
    };
    let inferred_ct = match category {
        FileCategory::Pdf => "application/pdf",
        FileCategory::Image => "image/*",
        FileCategory::Audio => "audio/*",
        FileCategory::Video => "video/*",
        FileCategory::Text => "text/plain",
        FileCategory::Html => "text/html",
        FileCategory::Office => "application/octet-stream",
        FileCategory::Unknown => "application/octet-stream",
    };
    Classification {
        filename: filename.to_string(),
        category,
        content_type: content_type
            .map(|s| s.to_string())
            .unwrap_or_else(|| inferred_ct.to_string()),
    }
}

/// Patterns matched (case-insensitive) when redacting sensitive map keys.
pub const SENSITIVE_PATTERNS: &[&str] =
    &["api_key", "password", "secret", "token", "credential"];

/// Best-effort redaction of values whose keys look sensitive.
pub fn redact_json(value: &mut serde_json::Value) {
    use serde_json::Value as V;
    match value {
        V::Object(map) => {
            for (k, v) in map.iter_mut() {
                let lk = k.to_ascii_lowercase();
                if SENSITIVE_PATTERNS.iter().any(|p| lk.contains(p)) {
                    if !v.is_null() {
                        *v = V::String("***REDACTED***".into());
                    }
                } else {
                    redact_json(v);
                }
            }
        }
        V::Array(arr) => {
            for v in arr.iter_mut() {
                redact_json(v);
            }
        }
        _ => {}
    }
}
