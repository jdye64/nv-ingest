# Use NVIDIA Morpheus as the base image
FROM nvcr.io/nvidia/morpheus/morpheus:23.11-runtime

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY src/ src/
COPY setup.py src/pipeline.py ./
RUN echo $(ls -la)
SHELL ["/bin/bash", "-c"]

# Install the module
RUN source activate morpheus \
    && pip install -e . \
    && rm -rf src

CMD ["python", "/workspace/pipeline.py"]
