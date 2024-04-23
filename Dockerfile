# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=morpheus-ms-base
ARG TAG=24.03

# Use NVIDIA Morpheus as the base image
FROM $BASE_IMG:$TAG as base

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY setup.py setup.py
# Don't copy full source here, pipelines won't be installed via setup anyway, and this allows us to rebuild more quickly if we're just changing the pipeline
COPY src/nv_ingest src/nv_ingest
COPY client nv_ingest_client
COPY requirements.txt test-requirements.txt util-requirements.txt ./
SHELL ["/bin/bash", "-c"]

# Prevent haystack from ending telemetry data
ENV HAYSTACK_TELEMETRY_ENABLED=False

RUN source activate morpheus \
    && pip install .

RUN source activate morpheus \
    && pip install ./nv_ingest_client \
    && rm -rf src requirements.txt test-requirements.txt util-requirements.txt

FROM base as runtime

COPY src/pipeline.py ./
COPY src/util/upload_to_ingest_ms.py ./
COPY pyproject.toml ./

CMD ["python", "/workspace/pipeline.py"]

FROM base as development

CMD ["/bin/bash"]
