# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger(__name__)


# Centralized client creation for handling retries and backoff
def create_inference_client(endpoint_url: str):
    """
    Create an inference client for communicating with a Triton server.

    Parameters
    ----------
    endpoint_url : str
        The URL of the Triton inference server.

    Returns
    -------
    grpcclient.InferenceServerClient
        A gRPC client for making inference requests to the Triton server.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> type(client)
    <class 'grpcclient.InferenceServerClient'>
    """
    return grpcclient.InferenceServerClient(url=endpoint_url)


def call_image_inference_model(client, model_name, image_data):
    inputs = [grpcclient.InferInput("input", image_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_data.astype(np.float32))

    outputs = [grpcclient.InferRequestedOutput("output")]

    try:
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        return " ".join([output[0].decode("utf-8") for output in result.as_numpy("output")])
    except Exception as e:
        logger.error(f"Inference failed for model {model_name}: {str(e)}")
        return None


# Perform inference and return predictions
def perform_model_inference(client, model_name: str, input_array: np.ndarray):
    """
    Perform inference using the provided model and input data.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        The gRPC client to use for inference.
    model_name : str
        The name of the model to use for inference.
    input_array : np.ndarray
        The input data to feed into the model, formatted as a numpy array.

    Returns
    -------
    np.ndarray
        The output of the model as a numpy array.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> input_array = np.random.rand(2, 3, 1024, 1024).astype(np.float32)
    >>> output = perform_model_inference(client, "my_model", input_array)
    >>> output.shape
    (2, 1000)
    """
    input_tensors = [grpcclient.InferInput("input", input_array.shape, datatype="FP32")]
    input_tensors[0].set_data_from_numpy(input_array)

    outputs = [grpcclient.InferRequestedOutput("output")]
    query_response = client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)
    logger.debug(query_response)

    return query_response.as_numpy("output")
