import mlflow
from typing import List, Tuple
from torch import nn
from ultralytics import YOLO


class YoloRegionProposal(nn.Module):
    def __init__(self, run_id: str):
        super().__init__()
        weights = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/weights/best.pt")
        self._model = YOLO(weights)

    def forward(self, image, threshold: float=0.5) -> List[Tuple]:
        result = self._model(image)[0]
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        mask = conf >= threshold
        boxes = boxes[mask]
        return boxes

if __name__ == "__main__":
    from hcmus.models.rpn import YoloRegionProposal
    run_id = "3ae25b8dcd324fbdbe8d5893f391e045"
    model = YoloRegionProposal(run_id)
    result = model("/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0b1403a927c13d2a9a0535901e12398f.jpg")
    result
    # boxes = result[0].boxes.xyxy
    # boxes[result[0].boxes.conf >= 0.5]
    # result[0].show()
