from typing import Dict, List
from pydantic import BaseModel, Field, field_validator
import base64

class Artifact(BaseModel):
    """
    Description: Base class for all artifacts.
    """
    asset_path: str = Field()
    version: str = Field()
    data: bytes = Field()
    metadata: Dict[str, str] = Field()
    type: str = Field()
    tags: List[str] = Field()

    @field_validator('type')
    def validate_type(cls, v):
        if ':' not in v:
            raise ValueError("Type must follow the format '<category>:<framework>'.")
        return v
