from pydantic import BaseModel


# Define a base class with extra fields forbidden
class BaseModelNoExt(BaseModel):
    class Config:
        extra = "forbid"
