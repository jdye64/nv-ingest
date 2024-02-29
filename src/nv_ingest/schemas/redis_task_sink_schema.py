from pydantic import BaseModel

from nv_ingest.schemas.redis_client_schema import RedisClientSchema


class RedisTaskSinkSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
