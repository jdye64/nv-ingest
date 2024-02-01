from typing import Optional, Literal

from pydantic import BaseModel, validator, conint


class DocumentSplitterSchema(BaseModel):
    split_by: Literal["word", "sentence", "passage"] = "word"
    split_length: conint(gt=0) = 200
    split_overlap: conint(ge=0) = 0
    max_character_length: Optional[conint(gt=0)] = None
    sentence_window_size: Optional[conint(ge=0)] = 0

    @validator('sentence_window_size')
    def check_sentence_window_size(cls, v, values, **kwargs):
        if v is not None and v > 0 and values['split_by'] != 'sentence':
            raise ValueError("When using sentence_window_size, split_by must be 'sentence'.")
        return v
