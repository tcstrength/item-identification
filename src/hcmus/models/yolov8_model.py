from typing import Optional, List, Tuple, Union
import torch
import cv2
import numpy as np
from pathlib import Path

class YOLOv8Model:
    """YOLOv8 model wrapper for object detection"""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> None:
        """
        Initialize YOLOv8 model.

        Args:
            model_path: Path to model weights
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model()

    def _load_model(self) -> Optional[torch.nn.Module]:
        """Load YOLOv8 model from path"""
        try:
            model = torch.hub.load('ultralytics/yolov8', 'custom',
                                 path=str(self.model_path),
                                 device=self.device)
            model.conf = self.conf_threshold
            model.iou = self.iou_threshold
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, image: np.ndarray) -> Optional[List[dict]]:
        """
        Run inference on image.

        Args:
            image: BGR image as numpy array

        Returns:
            List of detections, each a dict with keys:
            'bbox' (x1,y1,x2,y2), 'conf', 'class_id', 'class_name'
        """
        if self.model is None:
            return None

        try:
            results = self.model(image)
            detections = []

            # Process results
            for i, det in enumerate(results.xyxy[0]):
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(conf),
                    'class_id': int(cls),
                    'class_name': results.names[int(cls)]
                })

            return detections

        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def draw_predictions(
        self,
        image: np.ndarray,
        detections: List[dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on image.

        Args:
            image: BGR image to draw on
            detections: List of detections from predict()
            color: BGR color for boxes
            thickness: Line thickness

        Returns:
            Image with drawings
        """
        img = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['conf']:.2f}"

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1-text_size[1]-4), (x1+text_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        return img
