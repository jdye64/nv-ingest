# Use NVIDIA Morpheus as the base image
FROM devin-morpheus-ms:manual_0.01

# Set the working directory in the container
WORKDIR /workspace

# Copy the module code
COPY setup.py setup.py
COPY src/ src/
SHELL ["/bin/bash", "-c"]

# TODO(Devin): Something is wrong with the container build on the vdb_upload branch, its missing some of the module files.
# Double check this tomorrow.

# Install the module
RUN source activate morpheus \
    && mamba install -c conda-forge pyinstrument \
    && pip install redis==5.0.1 \
    && pip install . \
    && rm -rf src 

COPY src/pipeline.py ./

CMD ["python", "/workspace/pipeline.py"]
