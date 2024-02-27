from pydantic import BaseModel

from morpheus_pdf_ingest.schemas.redis_client_schema import RedisClientSchema


class RedisTaskSinkSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
