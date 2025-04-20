from typing import Dict, List, Any
from pydantic import BaseModel
from PIL.Image import Image


class ModelOutput(BaseModel):
    box: List[int]
    label: str
    metadata: Dict[str, Any]


class BasePredictor:
    def predict(self, image: Image) -> List[ModelOutput]:
        raise NotImplementedError()
