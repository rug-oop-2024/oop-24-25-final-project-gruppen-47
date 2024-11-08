from typing import Dict, List
from pydantic import BaseModel, Field, field_validator
import base64

class Artifact(BaseModel):
    """Description: Base class for all artifacts."""
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field(default="1.0.0")
    metadata: Dict[str, str] = Field(default=dict)
    type: str = Field(default="")
    tags: List[str] = Field(default=list)

    # @field_validator('type')
    # def validate_type(cls, v):
    #     if ':' not in v:
    #         raise ValueError("Type must follow the format '<category>:<framework>'.")
    #     return v
    
    @property
    def id(self) -> str:
        path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{path}:{self.version}"
    
    def read(self) -> bytes:
        return self.data
    
    def save(self, data: bytes) -> bytes:
        self.data = data
        return self.data