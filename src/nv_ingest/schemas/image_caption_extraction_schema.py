# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from pydantic import BaseModel


# TODO: Update to use Morpheus inference stage or Triton Client
class ImageCaptionExtractionSchema(BaseModel):
    batch_size: int = 8
    endpoint_url: str = "http://triton:8000/v2/models/caption_classification/infer"
    headers: dict = {"Content-Type": "application/json"}
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
