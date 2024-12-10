# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import logging
import os

from fastapi import (
    APIRouter,
    HTTPException,
    FastAPI,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    Response,
    status,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("uvicorn")

router = APIRouter()


# # Middleware for CORS (if needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# HTML page with dynamic table
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Metrics</title>
    <style>
        table {
            width: 50%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>Live Metrics</h1>
    <table id="metricsTable">
        <thead>
            <tr>
                <th>Key</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <!-- Dynamic rows will be appended here -->
        </tbody>
    </table>

    <script>
        const websocket = new WebSocket("ws://localhost:7670/v1/ws/metrics");

        websocket.onmessage = function(event) {
            const metrics = JSON.parse(event.data);
            const tableBody = document.querySelector("#metricsTable tbody");
            tableBody.innerHTML = ""; // Clear existing rows
            for (const [key, value] of Object.entries(metrics)) {
                const row = document.createElement("tr");
                const keyCell = document.createElement("td");
                const valueCell = document.createElement("td");
                keyCell.textContent = key;
                valueCell.textContent = value;
                row.appendChild(keyCell);
                row.appendChild(valueCell);
                tableBody.appendChild(row);
            }
        };

        websocket.onclose = function(event) {
            console.log("WebSocket closed:", event);
        };

        websocket.onerror = function(error) {
            console.error("WebSocket error:", error);
        };
    </script>
</body>
</html>
"""


@router.get(
    "/metrics",
    tags=["Metrics"],
    summary="Provides a summary of metrics information about the system",
    description="""
        Provide visibility into the running system and its performance.
    """,
    status_code=status.HTTP_200_OK,
    response_class=HTMLResponse
)
async def get_metrics():
    return html


# WebSocket endpoint
@router.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate dynamic metric updates (replace with actual data source)
            import random
            metrics = {
                "cpu_usage": f"{random.randint(1, 100)}%",
                "memory_usage": f"{random.randint(1, 100)}%",
                "disk_io": f"{random.randint(1, 100)} MB/s"
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(1)  # Update every second
    except WebSocketDisconnect:
        print("Client disconnected")
