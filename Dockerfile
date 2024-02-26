# Use NVIDIA Morpheus as the base image
FROM morpheus-ms-base:24.03 as base

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY setup.py setup.py
COPY src/ src/
SHELL ["/bin/bash", "-c"]

# Prevent haystack from ending telemetry data
ENV HAYSTACK_TELEMETRY_ENABLED=False

# Install the module
RUN source activate morpheus \
    && mamba install -c conda-forge \
      pydantic pyinstrument onnx=1.15.0 chardet=5.2.0 \
    && pip install redis==5.0.1 \
      PyMuPDF more_itertools \
      sentence_transformers==2.3.1 \
      unstructured-client==0.18.0 \
      farm-haystack[ocr,inference,pdf,preprocessing,file-conversion]

FROM base as runtime

RUN source activate morpheus \
    && pip install . \
    && rm -rf src 

COPY src/pipeline.py ./
COPY src/util/upload_to_ingest_ms.py ./

CMD ["python", "/workspace/pipeline.py"]

FROM base as development

CMD ["/bin/bash"]
