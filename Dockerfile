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
    && mamba install -c conda-forge pydantic pyinstrument \
    && pip install redis==5.0.1 PyMuPDF more_itertools\
    && pip install . \
    && rm -rf src 

COPY src/pipeline.py ./

CMD ["python", "/workspace/pipeline.py"]
