from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    Description: Base class for all features.

    Attributes:
        name (str): The name of the feature.
        type (Literal["categorical", "numerical"]): The type of the feature.
    """

    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        Returns:
            str: String representation of the feature.
        """
        return f"Feature(name={self.name}, type={self.type})"
