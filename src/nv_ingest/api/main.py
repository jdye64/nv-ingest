# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .v1.health import router as HealthApiRouter
from .v1.ingest import router as IngestApiRouter
from .v1.metrics import router as MetricsApiRouter
from .v1.configuration import router as ConfigurationApiRouter

logger = logging.getLogger(__name__)

# nv-ingest FastAPI app declaration
app = FastAPI()

app.include_router(IngestApiRouter)
app.include_router(HealthApiRouter)

# Set up the tracer provider and add a processor for exporting traces
resource = Resource(attributes={"service.name": "nv-ingest"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel-collector:4317")
exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

logger = logging.getLogger("uvicorn")

# nv-ingest FastAPI app declaration
app = FastAPI()

app.include_router(IngestApiRouter)
app.include_router(HealthApiRouter)
app.include_router(MetricsApiRouter)
app.include_router(ConfigurationApiRouter)

# Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "v1/static"), name="static")
