# Use NVIDIA Morpheus as the base image
FROM morpheus-ms-base:24.03 as base

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY setup.py setup.py
COPY src/ src/
COPY requirements.txt test-requirements.txt util-requirements.txt ./
SHELL ["/bin/bash", "-c"]

# Prevent haystack from ending telemetry data
ENV HAYSTACK_TELEMETRY_ENABLED=False

FROM base as runtime

RUN source activate morpheus \
    && pip install . \
    && rm -rf src requirements.txt test-requirements.txt util-requirements.txt

COPY src/pipeline.py ./
COPY src/util/upload_to_ingest_ms.py ./

CMD ["python", "/workspace/pipeline.py"]

FROM base as development

CMD ["/bin/bash"]
