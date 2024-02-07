from pydantic import BaseModel, ValidationError
from typing import Optional, Literal
from pydantic import conint, Field

class RedisTaskSourceSchema(BaseModel):
    redis_host: str = 'redis'
    redis_port: conint(gt=0, lt=65536) = 6379  # Ports must be in the range 1-65535
    task_queue: str = 'morpheus_task_queue'