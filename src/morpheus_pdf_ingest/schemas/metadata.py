from datetime import datetime
from enum import Enum
from typing import List, Union, Dict, Any

from morpheus_pdf_ingest.schemas.base_model_noext import BaseModelNoExt


## Do we want types and similar items to be enums or just strings?
class SourceTypeEnum(str, Enum):
    source_type_1 = 'source_type_1'
    source_type_2 = 'source_type_2'


class AccessLevelEnum(Enum):
    level_1 = 1
    level_2 = 2
    level_3 = 'level_3'


class ContentTypeEnum(str, Enum):
    type_1 = 'type_1'
    type_2 = 'type_2'


class TextTypeEnum(str, Enum):
    text_type_1 = 'text_type_1'
    text_type_2 = 'text_type_2'


class LanguageEnum(str, Enum):
    EN = 'English'
    FR = 'French'


class ImageTypeEnum(str, Enum):
    image_type_1 = 'image_type_1'
    image_type_2 = 'image_type_2'


# Sub schemas
class SourceMetadataSchema(BaseModelNoExt):
    source_name: str
    source_id: str
    source_location: str
    source_type: SourceTypeEnum
    collection_id: str
    date_created: datetime
    last_modified: datetime
    summary: str
    partition_id: int
    access_level: Union[AccessLevelEnum, int, str]


class ContentMetadataSchema(BaseModelNoExt):
    type: ContentTypeEnum
    description: str
    page_number: int
    hierarchy: Union[str, Dict]


class TextMetadataSchema(BaseModelNoExt):
    text_type: TextTypeEnum
    summary: str
    keywords: Union[str, List[str], Dict]
    language: LanguageEnum


class ImageMetadataSchema(BaseModelNoExt):
    image_type: ImageTypeEnum
    structured_image_type: ImageTypeEnum
    caption: str
    text: str
    image_location: str


# Main metadata schema
class MetadataSchema(BaseModelNoExt):
    content: str
    source_metadata: SourceMetadataSchema
    content_metadata: ContentMetadataSchema
    text_metadata: TextMetadataSchema
    image_metadata: ImageMetadataSchema


def validate_metadata(metadata: Dict[str, Any]) -> MetadataSchema:
    """
    Validates the given metadata dictionary against the MetadataSchema.

    Parameters:
    - metadata: A dictionary representing metadata to be validated.

    Returns:
    - An instance of MetadataSchema if validation is successful.

    Raises:
    - ValidationError: If the metadata does not conform to the schema.
    """
    return MetadataSchema(**metadata)
