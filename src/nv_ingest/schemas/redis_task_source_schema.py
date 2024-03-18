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

from nv_ingest.schemas.redis_client_schema import RedisClientSchema


class RedisTaskSourceSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
    task_queue: str = "morpheus_task_queue"
