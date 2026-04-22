# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricSummary(BaseModel):
    model_name: str
    invocation_count: int = 0
    pages_processed: int = 0
    detections_count: int = 0
    duration_ms: float = 0.0


class PageSummary(BaseModel):
    page_number: int
    content: dict[str, Any] = Field(default_factory=dict)


class IngestAccepted(BaseModel):
    document_id: str
    job_id: str | None = None
    content_sha256: str
    status: str
    created_at: str


class IngestStatus(BaseModel):
    document_id: str
    filename: str
    content_sha256: str
    status: str
    total_pages: int | None = None
    pages_received: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    created_at: str
    updated_at: str


class IngestComplete(BaseModel):
    document_id: str
    filename: str
    content_sha256: str
    status: str
    total_pages: int
    pages_received: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    created_at: str
    updated_at: str


class JobStatus(BaseModel):
    job_id: str
    filename: str
    status: str
    total_pages: int
    pages_submitted: int = 0
    pages_completed: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    created_at: str
    updated_at: str
