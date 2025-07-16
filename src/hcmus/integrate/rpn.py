import mlflow
from typing import List, Tuple
from torch import nn
from ultralytics import YOLO

class RegionProposal(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_iou(boxA, boxB):
        # Calculate intersection
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0.0

        # Calculate union
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxA_area + boxB_area - inter_area

        return inter_area / union_area

    @staticmethod
    def is_inside(inner, outer):
        return (inner[0] >= outer[0] and inner[1] >= outer[1] and
                inner[2] <= outer[2] and inner[3] <= outer[3])

    @staticmethod
    def remove_duplicates(boxes, conf, iou_threshold=0.9):
        # Pair boxes with confidence scores
        box_conf_pairs = list(zip(boxes, conf))
        # Sort by confidence descending
        box_conf_pairs.sort(key=lambda x: x[1], reverse=True)

        kept = []

        for box, score in box_conf_pairs:
            discard = False
            for kept_box, _ in kept:
                if RegionProposal.is_inside(box, kept_box):
                    discard = True
                    break
                if RegionProposal.compute_iou(box, kept_box) > iou_threshold:
                    discard = True
                    break
            if not discard:
                kept.append((box, score))
        if len(kept) == 0:
            return [], []
        boxes, scores = zip(*kept)  # This gives tuples
        boxes = list(boxes)
        scores = list(scores)
        return boxes, scores # List of (box, score)

    def forward(self, image, threshold: float=0.5) -> List[Tuple]:
        raise NotImplementedError

class YoloRegionProposal(RegionProposal):
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
        conf = conf[mask]
        return boxes, conf

if __name__ == "__main__":
    from hcmus.integrate.rpn import YoloRegionProposal
    run_id = "3ae25b8dcd324fbdbe8d5893f391e045"
    model = YoloRegionProposal(run_id)
    result = model("/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0b1403a927c13d2a9a0535901e12398f.jpg")
    result
    # boxes = result[0].boxes.xyxy
    # boxes[result[0].boxes.conf >= 0.5]
    # result[0].show()
