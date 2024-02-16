# Use NVIDIA Morpheus as the base image
FROM morpheus-ms-base:24.03

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY setup.py setup.py
COPY src/ src/
SHELL ["/bin/bash", "-c"]

# Install the module
RUN source activate morpheus \
    && mamba install -c conda-forge \
      pydantic pyinstrument onnx=1.15.0 \
    && pip install redis==5.0.1 \
      PyMuPDF more_itertools \
      sentence_transformers==2.3.1 \
      unstructured-client==0.18.0 \
      farm-haystack[all-gpu]==1.24.1 \
    && pip install . \
    && rm -rf src 

COPY src/pipeline.py ./
COPY src/util/upload_to_ingest_ms.py ./

CMD ["python", "/workspace/pipeline.py"]
