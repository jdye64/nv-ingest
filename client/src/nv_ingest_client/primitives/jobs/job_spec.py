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
import typing
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.primitives.tasks.extract import ExtractTask

logger = logging.getLogger(__name__)


class JobSpec:
    """
    Specification for creating a job for submission to the nv-ingest microservice.

    Parameters
    ----------
    payload : Dict
        The payload data for the job.
    tasks : Optional[List], optional
        A list of tasks to be added to the job, by default None.
    source_id : Optional[str], optional
        An identifier for the source of the job, by default None.
    job_id : Optional[UUID], optional
        A unique identifier for the job, by default a new UUID is generated.
    extended_options : Optional[Dict], optional
        Additional options for job processing, by default None.

    Attributes
    ----------
    _payload : Dict
        Storage for the payload data.
    _tasks : List
        Storage for the list of tasks.
    _source_id : str
        Storage for the source identifier.
    _job_id : UUID
        Storage for the job's unique identifier.
    _extended_options : Dict
        Storage for the additional options.

    Methods
    -------
    to_dict() -> Dict:
        Converts the job specification to a dictionary.
    add_task(task):
        Adds a task to the job specification.
    """

    def __init__(
        self,
        payload: str = None,
        tasks: Optional[List] = None,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
        document_type: Optional[str] = None,
        job_id: Optional[typing.Union[UUID, str]] = None,
        extended_options: Optional[Dict] = None,
    ) -> None:
        self._document_type = document_type or "txt"
        self._extended_options = extended_options or {}
        self._job_id = job_id
        self._payload = payload
        self._source_id = source_id
        self._source_name = source_name
        self._tasks = tasks or []

    def __str__(self) -> str:
        task_info = "\n".join(str(task) for task in self._tasks)
        return (
            f"job-id: {self._job_id}\n"
            f"source-id: {self._source_id}\n"
            f"task count: {len(self._tasks)}\n"
            f"{task_info}"
        )

    def to_dict(self) -> Dict:
        """
        Converts the job specification instance into a dictionary suitable for JSON serialization.

        Returns
        -------
        Dict
            A dictionary representation of the job specification.
        """
        # None of this is necessary with Pydantic ...
        if self._tasks and isinstance(self._tasks[0], ExtractTask):
            fixed_tasks = [task.to_dict() for task in self._tasks]
        else:
            fixed_tasks = self._tasks

        return {
            "job_payload": {
                "source_name": [self._source_name],
                "source_id": [self._source_id],
                "content": [self._payload],
                "document_type": [self._document_type],
            },
            "job_id": str(self._job_id),
            "tasks": fixed_tasks,
            "tracing_options": self._extended_options.get("tracing_options", {}),
        }

    @staticmethod
    def from_dict(d: dict):
        """
        Converts a Python Dict to a proper JobSpec object
        """
        print(f"Tasks Type: {type(d['tasks'])}")
        print(f"Tasks: {d['tasks']}")
        js = JobSpec(
            payload=d["job_payload"]["content"],
            tasks=Task.from_dict(d["tasks"]),
            source_id=d["job_payload"]["source_id"],
            source_name=d["job_payload"]["source_name"],
            document_type=d["job_payload"]["document_type"],
            job_id=d["job_id"],
            extended_options={"tracing_options": d["tracing_options"]},
        )
        return js

    @property
    def payload(self) -> Dict:
        return self._payload

    @payload.setter
    def payload(self, payload: Dict) -> None:
        self._payload = payload

    @property
    def job_id(self) -> UUID:
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: UUID) -> None:
        self._job_id = job_id

    @property
    def source_id(self) -> str:
        return self._source_id

    @source_id.setter
    def source_id(self, source_id: str) -> None:
        self._source_id = source_id

    @property
    def source_name(self) -> str:
        return self._source_name

    @source_name.setter
    def source_name(self, source_name: str) -> None:
        self._source_name = source_name

    def add_task(self, task) -> None:
        """
        Adds a task to the job specification.

        Parameters
        ----------
        task
            The task to add to the job specification. Assumes the task has a to_dict method.

        Raises
        ------
        ValueError
            If the task does not have a to_dict method.
        """
        if not isinstance(task, Task):
            raise ValueError("Task must derive from nv_ingest_client.primitives.Task class")

        self._tasks.append(task)
