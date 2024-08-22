# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
from typing import Any

from nv_ingest_client.client.client import NvIngestClient
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.util.util import check_ingest_result

from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest.util.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)


class RedisIngestService(IngestServiceMeta):
    """Submits Jobs to Morpheus via Redis"""

    _concurrency_level = os.getenv("CONCURRENCY_LEVEL", 10)
    _client_kwargs = "{}"
    __shared_instance = None

    _pending_jobs = []

    @staticmethod
    def getInstance():
        """Static Access Method"""
        if RedisIngestService.__shared_instance is None:
            redis_host = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
            redis_port = os.getenv("MESSAGE_CLIENT_PORT", "6379")
            redis_task_queue = os.getenv("REDIS_MORPHEUS_TASK_QUEUE", "morpheus_task_queue")
            RedisIngestService.__shared_instance = RedisIngestService(redis_host, redis_port, redis_task_queue)
        return RedisIngestService.__shared_instance

    def __init__(self, redis_hostname: str, redis_port: int, redis_task_queue: str):
        self._redis_hostname = redis_hostname
        self._redis_port = redis_port
        self._redis_task_queue = redis_task_queue

        # Create the ingest client
        self._ingest_client = NvIngestClient(
            message_client_allocator=RedisClient,
            message_client_hostname=self._redis_hostname,
            message_client_port=self._redis_port,
            worker_pool_size=self._concurrency_level,
        )

    async def submit_job(self, job_spec: JobSpec) -> str:
        try:
            job_id = self._ingest_client.add_job(job_spec)
            print(f"Type of JobSpec: {type(job_spec)}")
            print(f"&&&& JobSpec: '{job_spec}'")
            # [({'status': 'failed', 'description': 'Failed to process the message.', 'data': None, 'trace': {}, 'annotations': {'annotation::2c64e2dc-3c7b-403c-a2fd-c1143e898798': {'error': '6 validation errors for IngestJobSchema\njob_payload -> content -> 0\n  str type expected (type=type_error.str)\njob_payload -> content -> 0\n  byte type expected (type=type_error.bytes)\njob_payload -> source_name -> 0\n  str type expected (type=type_error.str)\njob_payload -> source_id -> 0\n  str type expected (type=type_error.str)\njob_payload -> source_id -> 0\n  value is not a valid integer (type=type_error.integer)\njob_payload -> document_type -> 0\n  str type expected (type=type_error.str)', 'message': 'Failed to process job submission', 'source_id': 'nv_ingest.modules.sources.redis_task_source'}, 'annotation::4797b6ed-6391-4914-a555-371cdcd7d3b0': {'message': "'NoneType' object has no attribute 'mutable_dataframe'", 'source_id': 'nv_ingest.util.exception_handlers.decorators', 'task_id': 'job_counter', 'task_result': 'FAILURE'}}}, 'ced65225-2705-4554-97e1-4b30ee47e0f4')]
            breakpoint()
            _ = self._ingest_client.submit_job(job_id, self._redis_task_queue)
            self._pending_jobs.extend(job_id)
            return job_id
        except Exception as err:
            logger.error("Error: %s", err)
            raise

    async def fetch_job(self, job_id: str) -> Any:
        print(f"Before fetch_job_result_async")
        futures_dict = self._ingest_client.fetch_job_result_async(job_id, timeout=60, data_only=False)
        breakpoint()
        print(f"After fetch_job_result_async")
        futures = list(futures_dict.keys())
        result = futures[0].result()
        if len(result) > 1:
            logger.error(f"Need to figure out what is going on here .... {result} len: {len(result)}")
        result = result[0]  # List, get first element
        result = result[0]  # Tuple (response, job_id), get response
        if ("annotations" in result) and result["annotations"]:
            annotations = result["annotations"]
            for key, value in annotations.items():
                logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

        is_failed, description = check_ingest_result(result)

        if is_failed:
            breakpoint()
            raise RuntimeError(f"Failed to process job {job_id}: {description}")

        return result

    # def get_jobs(self) -> Any:
    #     """Retrieve the results from submitted jobs"""
    #     results = []
    #     for job_id in self._pending_jobs:
    #         result = self._ingest_client.fetch_job_result(job_id, data_only=False)
    #         results.append(result)
    #         self._pending_jobs.remove(job_id)
    #     return results
