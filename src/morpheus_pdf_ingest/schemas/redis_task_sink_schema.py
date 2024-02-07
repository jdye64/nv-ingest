from pydantic import BaseModel, ValidationError
from typing import Optional
from pydantic import conint

class RedisTaskSinkSchema(BaseModel):
    redis_host: str = 'redis'
    redis_port: conint(gt=0, lt=65536) = 6379  # Validate port is in a valid range

    # Add additional fields if the module has more configurable parameters
