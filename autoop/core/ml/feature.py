from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"
