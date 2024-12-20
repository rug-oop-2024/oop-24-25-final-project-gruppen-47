from typing import Dict, List
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """Description: Base class for all artifacts."""

    name: str = Field()
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field(default="1.0.0")
    metadata: Dict[str, str] = Field(default_factory=dict)
    type: str = Field(default="")
    tags: List[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """
        Returns the unique identifier for the artifact

        Returns:
            str: Unique identifier
        """
        path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{path}:{self.version}"

    def read(self) -> bytes:
        """
        Reads the artifact"

        Returns:
            bytes: Artifact data
        """
        return self.data

    def save(self, data: bytes) -> bytes:
        """
        Saves the artifact

        Args:
            data (bytes): Data to save
        Returns:
            bytes: Saved data
        """
        self.data = data
        return self.data
