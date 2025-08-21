import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple
from hcmus.models.backbone import BaseBackbone


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        dropout: float = 0.2,
        feature_dim: int = 512
    ):
        super().__init__()

        self._backbone = backbone
        self._feature_dim = feature_dim
        self._projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._backbone.output_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

    def encode_images(self, images: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        features = self._backbone(images)
        features = self._projector(features.float())
        if not normalized:
            return features

        return F.normalize(features, dim=-1)

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        n_classes = len(support_labels.unique())

        prototypes = torch.zeros(
            n_classes, self._feature_dim,
            device=support_features.device
        )

        for c in range(n_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                prototypes[c] = support_features[class_mask].mean(dim=0)

        return prototypes

    def forward(
        self,
        support_images: torch.Tensor = None,
        support_labels: torch.Tensor = None,
        query_images: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support_features = self.encode_images(support_images)
        query_features = self.encode_images(query_images)
        prototypes = self.compute_prototypes(support_features, support_labels)
        distances = torch.cdist(query_features, prototypes)
        logits = -distances
        return logits, prototypes

class PrototypicalTrainer:
    def __init__(self, model, optimizer, criterion_fn):
        self.model = model
        self.optimizer = optimizer
        self.criterion_fn = criterion_fn

    def train_episode(self, support_data, support_labels, query_data, query_labels):
        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self.model(support_data, support_labels, query_data)

        loss = self.criterion_fn(logits, query_labels)

        loss.backward()
        self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels).float().mean().item()

        return loss.item(), acc

    def evaluate_episode(self, support_data, support_labels, query_data, query_labels):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(support_data, support_labels, query_data)
            loss = self.criterion_fn(logits, query_labels)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == query_labels).float().mean().item()
        return loss.item(), acc

if __name__ == "__main__":
    from hcmus.utils import data_utils
    from hcmus.utils import transform_utils
    from hcmus.models.backbone import CLIPBackbone
    splits = data_utils.get_data_splits(["train"])
    transforms_train, transforms_test = transform_utils.get_transforms_downscale_random_v2()
    train_dataset = data_utils.get_image_datasets_v2(splits, transform_train=transforms_train)["train"]
    device = "mps"
    backbone = CLIPBackbone("ViT-B/32", device="mps")
    model = PrototypicalNetwork(backbone)
    model.to(device)
    image, label, _ = train_dataset[0]

    features = backbone(image.half().to(device))
    features.shape

