# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from fastapi import FastAPI
from fastapi import status
import os

# # Remote debugging
# import debugpy

from .api.main import app as app_v1

# # Remote debugging configuration
# print(f'REMOTE_DEBUG_ENABLED value: {os.getenv("REMOTE_DEBUG_ENABLED")}')
# if os.getenv("REMOTE_DEBUG_ENABLED", "").lower() in ['true', '1', 'yes', 'on']:
#     # Set the host and port for remote debugger
#     debugpy.listen(("0.0.0.0", 5678))
#     print("Waiting for debugger to attach on port 5678 ...")
#     debugpy.wait_for_client()
#     print("Debugger attached. Continuing execution...")
# else:
#     print("Remote debugging is not enabled")

app = FastAPI(
    title="NV-Ingest Microservice",
    description="Service for ingesting heterogenous datatypes",
    version="0.1.0",
    contact={
        "name": "NVIDIA Corporation",
        "url": "https://nvidia.com",
    },
    openapi_tags=[
        {"name": "Health", "description": "Health checks"},
    ],
)


app.mount("/v1", app_v1)


@app.get(
    "/health",
    tags=["Health"],
    summary="Perform a Health Check",
    description="""
        Immediately returns 200 when service is up.
        This does not check the health of downstream
        services.
    """,
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_health() -> str:
    # Perform a health check
    return "OK"
