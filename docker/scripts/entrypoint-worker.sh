#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash

set -e

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && . "${SRC_FILE}"

SRC_EXT="/workspace/docker/entrypoint_source_ext.sh"
[ -f "${SRC_EXT}" ] && . "${SRC_EXT}"

# Determine ingest config path, if exists and is a valid file.
if [ -n "${INGEST_CONFIG_PATH}" ] && [ -f "${INGEST_CONFIG_PATH}" ]; then
    CONFIG_ARG="--ingest_config_path=${INGEST_CONFIG_PATH}"
else
    CONFIG_ARG=""
fi

# Check if INGEST_MEM_TRACE is set to 1, true, on, or yes (case-insensitive)
MEM_TRACE=false
if [ -n "${INGEST_MEM_TRACE}" ]; then
    case "$(echo "${INGEST_MEM_TRACE}" | tr '[:upper:]' '[:lower:]')" in
        1|true|on|yes)
            MEM_TRACE=true
            ;;
    esac
fi

# Check if user supplied a command
if [ "$#" -gt 0 ]; then
    # If a command is provided, run it.
    exec "$@"
else
    # --- Launch the worker ---
    celery -A nv_ingest.api.v1.celery_worker worker --loglevel=info
fi
