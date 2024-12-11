# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: skip-file

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from opentelemetry import trace

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()

# Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
router.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# GET /fetch_job
@router.get(
    "/configuration",
    responses={
        200: {"description": "Job was successfully retrieved."}
    },
    tags=["Configuration"],
    summary="Gather the configurations from the current running instance.",
    operation_id="fetch_job",
    response_class=HTMLResponse
)
async def gather_configurations(request: Request):
    env_vars = dict(os.environ)
    sorted_env_vars = dict(sorted(env_vars.items()))
    del sorted_env_vars['NGC_API_KEY']
    del sorted_env_vars['NVIDIA_BUILD_API_KEY']

    # Check the Accept header
    accept = request.headers.get("accept", "")

    if "text/html" in accept:
        # Return HTML response
        return templates.TemplateResponse("configurations.html", {"request": request, "env_vars": sorted_env_vars})
    else:
        # Default to JSON response
        return JSONResponse(content=sorted_env_vars)
