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
COPY requirements.txt test-requirements.txt util-requirements.txt ./
SHELL ["/bin/bash", "-c"]

# Prevent haystack from ending telemetry data
ENV HAYSTACK_TELEMETRY_ENABLED=False

RUN source activate morpheus \
    && pip install . \
    && rm -rf src requirements.txt test-requirements.txt util-requirements.txt

FROM base as runtime

COPY src/pipeline.py ./
COPY src/util/upload_to_ingest_ms.py ./

CMD ["python", "/workspace/pipeline.py"]

FROM base as development

CMD ["/bin/bash"]
