# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

from fastapi import APIRouter
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import psutil
import threading
import time

router = APIRouter()

cpu_usage_histogram = Histogram("system_cpu_usage_histogram", "CPU usage over time")
memory_usage_histogram = Histogram("system_memory_usage_histogram", "Memory usage over time")


def collect_metrics():
    """Collects system metrics and updates Histograms periodically."""
    while True:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().used

        cpu_usage_histogram.observe(cpu_usage)
        memory_usage_histogram.observe(memory_usage)

        time.sleep(5)


thread = threading.Thread(target=collect_metrics, daemon=True)
thread.start()


@router.get("/metrics")
def get_metrics():
    """Expose Prometheus metrics via FastAPI."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
