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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from nv_ingest_client.primitives.tasks import Task
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class JobSpec(BaseModel):
    """
    Specification for creating a job for submission to the nv-ingest microservice.

    Attributes
    ----------
    document_type: str
        Type of document that is being submitted.
    extended_options : Dict
        Storage for the additional options.
    job_id : UUID
        Storage for the job's unique identifier.
    payload : str
        Storage for the payload data.
    source_id : str
        Storage for the source identifier.
    source_name: str
        Storage for the source name.
    tasks : List
        Storage for the list of tasks.

    Methods
    -------
    add_task(task):
        Adds a task to the job specification.
    """

    _document_type: Optional[str] = Field(default=None, alias="document_type")
    _extended_options: Optional[Dict] = Field(default={}, alias="extended_options")
    _job_id: Optional[typing.Union[UUID, str]] = Field(default=None, alias="job_id")
    _payload: str = Field(default=None, alias="payload")
    _source_id: Optional[str] = Field(default=None, alias="source_id")
    _source_name: Optional[str] = Field(default=None, alias="source_name")
    _tasks: Optional[List] = Field(default=[], alias="tasks")

    def __str__(self) -> str:
        task_info = "\n".join(str(task) for task in self.tasks)
        return (
            f"job-id: {self.job_id}\n"
            f"source-id: {self.source_id}\n"
            f"source-name: {self.source_name}\n"
            f"document-type: {self.document_type}\n"
            f"task count: {len(self.tasks)}\n"
            f"payload: {'<*** ' + str(len(self.payload)) + ' ***>' if self.payload else 'Empty'}\n"
            f"extended-options: {self._xtended_options}\n"
            f"{task_info}"
        )

    @property
    def tasks(self) -> Any:
        """Gets the job specification associated with the state."""
        return self._tasks

    # @tasks.setter
    # def job_spec(self, value: JobSpec) -> None:
    #     """Sets the job specification associated with the state."""
    #     if self._state not in _PREFLIGHT_STATES:
    #         err_msg = f"Attempt to change job_spec after job submission: {self._state.name}"
    #         logger.error(err_msg)

    #         raise ValueError(err_msg)

    #     self._job_spec = value

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

        self.tasks.append(task)
