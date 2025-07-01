from IPython import embed
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Tuple, Optional

class CLIPBackbone(nn.Module):
    """
    ViT-B/16 backbone using CLIP's pre-trained model
    """
    def __init__(self, backbone_name: str = "ViT-B/32", freeze_backbone: bool = False, mlflow_logged_model: str = None):
        super().__init__()
        if mlflow_logged_model is not None:
            self.model = mlflow.pyfunc.load_model(mlflow_logged_model).get_raw_model()
        else:
            self.model, _ = clip.load(backbone_name, device="cuda" if torch.cuda.is_available() else "cpu")

        # Extract the visual encoder
        self.visual_encoder = self.model.visual

        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        # Feature dimension for ViT-B/16
        if "ViT-B" in backbone_name:
            self.feature_dim = 512
        elif "ViT-L" in backbone_name:
            self.feature_dim = 768
        elif "RN50" in backbone_name:
            self.feature_dim = 1024
        elif "RN101" in backbone_name:
            self.feature_dim = 512


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Feature tensor of shape (batch_size, 512)
        """
        return self.visual_encoder(x)


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        clip_model_name: str = 'ViT-B/32',
        feature_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_clip: bool = False
    ):
        super().__init__()

        self.clip_model, self.preprocess = clip.load(clip_model_name)

        clip_dim = self.clip_model.visual.output_dim

        if feature_dim is None:
            feature_dim = clip_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(clip_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.feature_dim = feature_dim

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if hasattr(self, '_freeze_clip') else torch.enable_grad():
            clip_features = self.clip_model.encode_image(images)

        features = self.feature_projector(clip_features.float())
        return F.normalize(features, dim=-1)

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:

        prototypes = torch.zeros(
            n_classes, self.feature_dim,
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
        query_images: torch.Tensor = None,
        n_classes: int = None,
        prototypes: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prototypes is None:
            support_features = self.encode_images(support_images)
            query_features = self.encode_images(query_images)
            prototypes = self.compute_prototypes(support_features, support_labels, n_classes)
        else:
            query_features = self.encode_images(query_images)

        distances = torch.cdist(query_features, prototypes)
        logits = -distances

        return logits, prototypes


class PrototypeTracker:
    def __init__(self, momentum=0.9):
        self.prototypes = {}
        self.momentum = momentum

    def update(self, embeddings, labels):
        """Update prototypes with exponential moving average"""
        # print(embeddings.shape, labels.shape)
        for emb, label_item in zip(embeddings, labels):
            # print(emb.shape)
            if label_item not in self.prototypes:
                self.prototypes[label_item] = emb.clone().detach()
            else:
                self.prototypes[label_item] = (
                    self.momentum * self.prototypes[label_item] +
                    (1 - self.momentum) * emb.detach()
                )

    def get_prototypes(self, classes: list[int]):
        prototypes = []
        for c in classes:
            prototypes.append(self.prototypes[c])
        return torch.vstack(prototypes)

