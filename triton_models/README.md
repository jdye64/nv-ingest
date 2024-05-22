## Table of Contents

- [Overview](#overview)
- [General Prerequisites](#general-prerequisites)
- [Model Configurations](#model-configurations)
  - [DeBerta Caption Selection Model](#deberta-caption-selection-model)
  - [ECLAIR](#eclair-document-ocr-model)
  - [E5-small-v2](#e5-small-v2)
- [Verifying Model Deployment](#verifying-model-deployment)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Additional Resources](#additional-resources)

## Overview

This README provides specific instructions for constructing the Triton model directory for multiple models. Each section
focuses on the requirements for a particular model, including configuration files, model weights, and special backend
handling.

## General Prerequisites

Pull the Triton container and extend it with some common libraries.

`Dockerfile.caption`

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.04-py3

RUN pip install pandas torch transformers
```

```bash
docker build --file Dockerfile.caption ./
```

## Model Configurations

For our purposes we'll assume that the Triton service is launched with a local directory, `models`, volume mounted for
Triton to use.

### DeBerta Caption Selection Model

#### Pre-requisites

We'll use the [ngc-cli](https://org.ngc.nvidia.com/setup/installers/cli) tool to download the model.

#### Model acquisition

```bash
ngc registry model download-version "nvidian/nemo-llm/nemo-retriever-caption-classification-triton-pytorch:1" \
  --dest ./

tar -xvf 1.tar.gz -C ./models
```

#### model.py File

Copy the `model.py` file from ./caption_classification/model.py to ./models/caption_classification/1/model.py

```bash
cp ./caption_classification/model.py ./models/caption_classification/1/model.py
```

#### .pbtxt Configuration

Create the following in ./models/caption_classification/config.pbtxt

```plaintext
name: "caption_classification"
backend: "python"
max_batch_size: 256

input [
  {
    name: "candidates"
    data_type: TYPE_STRING
    dims: [-1]
  }
]

output [
  {
    name: "caption"
    data_type: TYPE_STRING
    dims: [1]
  }
]

parameters: {
  key: "threshold"
  value: {
    string_value: "0.5"
  }
}

parameters: {
  key: "min_words"
  value:{
    string_value: "5"
  }
}

instance_group [
  {
    kind: KIND_AUTO
  }
]
```

#### Verifying unpackaing

```bash
# Model tree after unpacking
./models/
└── caption_classification
    ├── 1
    │   ├── model.py
    │   ├── __pycache__
    │   │   └── model.cpython-310.pyc
    │   ├── requirements.txt
    │   └── weights
    │       ├── caption_classification
    │       └── 1
    │           └── weights
    │               ├── fold0
    │               │   └── model
    │               │       └── model_0.pt
    │               ├── fold1
    │               │   └── model
    │               │       └── model_1.pt
    │               ├── fold2
    │               │   └── model
    │               │       └── model_2.pt
    │               ├── fold3
    │               │   └── model
    │               │       └── model_3.pt
    │               └── fold4
    │                   └── model
    │                       └── model_4.pt
    └── config.pbtxt
```

### ECLAIR Document OCR Model

In order to work around the limitations of GitLab CI/CD, relative URLs are used in `.gitmodules`:

```
[submodule "third_party/eclair_triton"]
	path = third_party/eclair_triton
	url = ../../../../edwardk/eclair-triton.git
	ignore = untracked
```

However, when you clone the repository locally, Git cannot deduce the URL correctly, so replace
the relative in `.gitmodules` with an absolute URL:

```
[submodule "third_party/eclair_triton"]
	path = third_party/eclair_triton
	url = ssh://git@gitlab-master.nvidia.com:12051/edwardk/eclair-triton.git
	ignore = untracked
```

After replacing `url` with the aboluste URL, checkout out the submodule:

```
git submodule update --init --recursive
```

The `eclair_triton` repository also hosts the model checkpoints as Git LFS files, so be sure to
pull the model weights as well:

```
git -C third_party/eclair_triton lfs pull
```

Using `--extract_method eclair` requires that there is a Triton server running.
To set up Triton, set the following environment variables in `.env`:

```
ECLAIR_CHECKPOINT_DIR=./third_party/eclair_triton/checkpoints
ECLAIR_CHECKPOINT_NAME=sweep_0_cooperative-swine_2024.04.20_22.35
ECLAIR_MODEL_DIR=./third_party/eclair_triton/models
ECLAIR_BATCH_SIZE=16
```

You may need to adjust the batch size depending on the GPU type.
The default batch size of 16 was optimized for A100.

![](../docs/images/eclair_batch_size.png)

First, build the base image:

```
docker compose -f third_party/eclair_triton/docker-compose.yaml build triton-trt-llm
```

Next, run the following to build a TensorRT model:

```
docker compose -f third_party/eclair_triton/docker-compose.yaml up build-eclair
```

Then, run the server:

```
docker compose -f docker-compose.yaml -f third_party/eclair_triton/docker-compose.yaml up triton-eclair
```

### E5-Small-V2
