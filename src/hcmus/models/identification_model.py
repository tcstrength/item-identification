import torch
import torchvision.transforms as T
from pydantic import BaseModel
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ModelResult(BaseModel):
    box: ObjectBox
    score: float
    label: int


class IdentificationModel:
    def __init__(self):
        self._model = fasterrcnn_resnet50_fpn(pretrained=True)
        self._model.eval()
        self._transform_compose = T.Compose([T.ToTensor()])

    def transform(self, image: Image) -> torch.Tensor:
        image_tensor = self._transform_compose(image).unsqueeze(0)
        return image_tensor

    def predict(self, tensor: torch.Tensor) -> list[ModelResult]:
        predictions = self._model(tensor)
        results = []
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            item = ModelResult(
                box=ObjectBox(
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
                ),
                score=score,
                label=label
            )
            results.append(item)
        return results
