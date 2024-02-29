import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskInjectionSchema(BaseModel):
    dummy_var: str = "dummy"

    class Config:
        extra = "forbid"
