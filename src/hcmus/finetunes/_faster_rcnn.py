import mlflow
import torch
from typing import Dict
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNWrapper:
    def __init__(
        self, num_classes: int,
        device: str = "cpu"
    ):
        self._model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self._model_params = [p for p in self._model.parameters() if p.requires_grad]
        self._device = device
        self._model.to(self._device)

        logger.info(f"Use torch device: {device}")

    def train(
        self,
        train_loader: DataLoader,
        hyper_params: Dict[str, float]
    ):
        optimizer = optim.SGD(
            self._model_params,
            lr=hyper_params.get("lr"),
            momentum=hyper_params.get("momentum"),
            weight_decay=hyper_params.get("weight_decay")
        )
        num_epochs = hyper_params.get("num_epochs")
        best_loss = float("inf")

        with mlflow.start_run():
            mlflow.log_params(hyper_params)

            for epoch in range(num_epochs):
                self._model.train()
                running_loss = 0.0

                for images, targets in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    images = [img.to(self._device) for img in images]
                    targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

                    loss_dict = self._model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                logger.info(f"Epoch {epoch+1}/{num_epochs} - Training loss: {running_loss:.4f}")

                if running_loss < best_loss:
                    best_loss = running_loss
                    mlflow.log_metrics({
                        "train_loss": running_loss
                    })
                    mlflow.pytorch.log_model(self._model, "model")
