# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

# The easiest way to run this script without having to install nv-ingest is adjust
# your PYTHONPATH to include the NV_INGEST_REPO_ROOT/src directory like so ...
# This script is intended to only be ran from NV_INGEST_REPO_ROOT
# PYTHONPATH=$(pwd)/src:$(pwd)/client/src scripts/generated_openapi_docs.py

import json
import os
from nv_ingest.main import app

# Define output directory and file
OUTPUT_DIR = "./docs/sphinx_docs/source/user_guide/openapi_docs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "openapi.json")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate OpenAPI schema
openapi_schema = app.openapi()

# Save OpenAPI schema to JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(openapi_schema, f, indent=2)

print(f"✅ OpenAPI schema saved to {OUTPUT_FILE}")
