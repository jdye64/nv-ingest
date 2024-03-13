from pydantic import BaseModel

from nv_ingest.schemas.redis_client_schema import RedisClientSchema


class RedisTaskSourceSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
    task_queue: str = "morpheus_task_queue"
