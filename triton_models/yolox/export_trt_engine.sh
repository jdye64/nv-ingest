#!/bin/bash

set -e

# build container
docker build -f docker/Dockerfile.yolox -t yolox_trt_export:24.06-py3 .
# run container to export the trt engine
docker run --gpus all -it --rm -v $PWD:/yolox -w /yolox  yolox_trt_export:24.06-py3 python3 generate_trt_engine.py
