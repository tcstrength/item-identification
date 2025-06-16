import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    ViT-B/16 backbone using CLIP's pre-trained model
    """
    def __init__(self, backbone_name: str = "ViT-B/16", freeze_backbone: bool = True):
        super().__init__()
        self.model, self.preprocess = clip.load(backbone_name, device="cuda" if torch.cuda.is_available() else "cpu")

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

class OpenWorldPrototypicalNetwork(nn.Module):
    """
    Prototypical Network with Open World capabilities using ViT-B/16
    """
    def __init__(
        self,
        backbone: nn.Module,
        temperature: float = 1.0,
        unknown_threshold: float = 0.5,
        use_cosine_similarity: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.temperature = temperature
        self.unknown_threshold = unknown_threshold
        self.use_cosine_similarity = use_cosine_similarity

        # Learnable temperature parameter
        self.learnable_temp = nn.Parameter(torch.tensor(temperature))

        # Optional projection head for fine-tuning
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set
        Args:
            support_features: (n_support, feature_dim)
            support_labels: (n_support,)
        Returns:
            prototypes: (n_classes, feature_dim)
        """
        n_classes = support_labels.max().item() + 1
        prototypes = torch.zeros(n_classes, self.feature_dim, device=support_features.device)

        for class_id in range(n_classes):
            class_mask = support_labels == class_id
            if class_mask.sum() > 0:
                prototypes[class_id] = support_features[class_mask].mean(dim=0)

        return prototypes

    def compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between queries and prototypes
        """
        if self.use_cosine_similarity:
            # Normalize features and prototypes
            query_norm = F.normalize(query_features, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)

            # Compute cosine similarity (convert to distance)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        else:
            # Euclidean distance
            distances = torch.cdist(query_features, prototypes, p=2)

        return distances

    def detect_unknown(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Detect unknown/novel classes based on distance threshold
        Args:
            distances: (n_query, n_classes)
        Returns:
            unknown_mask: (n_query,) boolean tensor
        """
        min_distances = distances.min(dim=1)[0]
        unknown_mask = min_distances > self.unknown_threshold
        return unknown_mask

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass for few-shot classification with open-world detection
        """
        # Extract features
        support_features = self.backbone(support_images)
        query_features = self.backbone(query_images)

        # Apply projection head
        support_features = self.projection_head(support_features)
        query_features = self.projection_head(query_features)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)

        # Compute distances
        distances = self.compute_distances(query_features, prototypes)

        # Convert distances to logits
        logits = -distances / self.learnable_temp

        # Detect unknown samples
        unknown_mask = self.detect_unknown(distances)

        # Compute probabilities
        probs = F.softmax(logits, dim=1)

        # Apply unknown detection (set probabilities to uniform for unknown samples)
        n_classes = prototypes.shape[0]
        unknown_prob = 1.0 / (n_classes + 1)  # +1 for unknown class

        # Create final predictions including unknown class
        final_probs = torch.zeros(query_features.shape[0], n_classes + 1, device=query_features.device)
        final_probs[:, :-1] = probs * (~unknown_mask).float().unsqueeze(1)
        final_probs[:, -1] = unknown_mask.float()

        # Normalize probabilities
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)

        results = {
            'logits': logits,
            'probabilities': final_probs,
            'distances': distances,
            'unknown_mask': unknown_mask,
            'prototypes': prototypes
        }

        if return_features:
            results['support_features'] = support_features
            results['query_features'] = query_features

        return results

class OpenWorldTrainer:
    """
    Trainer for Open World Prototypical Networks
    """
    def __init__(
        self,
        model: OpenWorldPrototypicalNetwork,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        unknown_mask: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Compute combined loss for known and unknown samples
        """
        # Classification loss for known samples
        known_mask = ~unknown_mask
        if known_mask.sum() > 0:
            ce_loss = F.cross_entropy(logits[known_mask], labels[known_mask])
        else:
            ce_loss = torch.tensor(0.0, device=self.device)

        # Unknown detection loss (encourage high distances for unknown samples)
        # This is a simplified version - you might want to use more sophisticated losses
        unknown_loss = torch.tensor(0.0, device=self.device)

        total_loss = alpha * ce_loss + (1 - alpha) * unknown_loss
        return total_loss

    def train_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        Train on a single episode
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        results = self.model(support_images, support_labels, query_images)

        # Compute loss (assuming all query samples are known for now)
        unknown_mask = torch.zeros(query_labels.shape[0], dtype=torch.bool, device=self.device)
        loss = self.compute_loss(results['logits'], query_labels, unknown_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predictions = results['probabilities'][:, :-1].argmax(dim=1)  # Exclude unknown class
        accuracy = (predictions == query_labels).float().mean()

        return {
            'train_loss': loss.item(),
            'train_accuracy': accuracy.item()
        }

    def evaluate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        has_unknown: bool = False
    ) -> dict:
        """
        Evaluate on a single episode
        """
        self.model.eval()

        with torch.no_grad():
            results = self.model(support_images, support_labels, query_images)

            # Get predictions (including unknown class)
            predictions = results['probabilities'].argmax(dim=1)
            n_classes = support_labels.max().item() + 1

            # Compute metrics
            if has_unknown:
                unknown_class_id = n_classes

                known_mask = (query_labels >= 0)
                unknown_mask = (query_labels == -1)
                # Calculate accuracy for known classes
                known_queries = predictions[known_mask]
                known_true_labels = query_labels[known_mask]
                known_accuracy = (known_queries == known_true_labels).float().mean() if known_mask.sum() > 0 else 0.0

                # Calculate unknown detection metrics
                unknown_queries = predictions[unknown_mask]
                unknown_detected_correctly = (unknown_queries == unknown_class_id).sum().item()
                total_unknown = unknown_mask.sum().item()
                unknown_recall = unknown_detected_correctly / total_unknown if total_unknown > 0 else 0.0

                # Calculate false positive rate (known samples predicted as unknown)
                known_predicted_as_unknown = (predictions[known_mask] == unknown_class_id).sum().item()
                total_known = known_mask.sum().item()
                false_positive_rate = known_predicted_as_unknown / total_known if total_known > 0 else 0.0

                # Overall accuracy including unknown detection
                correct_known = (predictions[known_mask] == query_labels[known_mask]).sum().item()
                correct_unknown = unknown_detected_correctly
                total_queries = len(query_labels)

                overall_accuracy = (correct_known + correct_unknown) / total_queries

                return {
                    'val_accuracy': overall_accuracy,
                    'known_accuracy': known_accuracy,
                    'unknown_recall': unknown_recall,
                    'false_positive_rate': false_positive_rate,
                    # 'predictions': predictions,
                    # 'unknown_detected': (predictions == n_classes).sum().item()
                }

            else:
                # Standard few-shot evaluation
                known_predictions = predictions[predictions < n_classes]
                known_labels = query_labels[predictions < n_classes]
                accuracy = (known_predictions == known_labels).float().mean() if len(known_predictions) > 0 else 0.0

                return {
                    'accuracy': accuracy.item(),  # Placeholder
                    # 'predictions': predictions,
                    # 'unknown_detected': (predictions == n_classes).sum().item()
                }

# Usage example
def create_model(backbone_name, freeze_backbone: bool):
    """
    Create the complete model
    """
    # Initialize backbone
    backbone = CLIPViTBackbone(backbone_name, freeze_backbone=freeze_backbone)

    # Create open-world prototypical network
    model = OpenWorldPrototypicalNetwork(
        backbone=backbone,
        temperature=1.0,
        unknown_threshold=0.6,
        use_cosine_similarity=True
    )

    return model

def setup_training(backbone_name, freeze_backbone: bool):
    """
    Setup training configuration
    """
    model = create_model(backbone_name, freeze_backbone)
    trainer = OpenWorldTrainer(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    return model, trainer, optimizer, scheduler

if __name__ == "__main__":
    # Example usage
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Example forward pass
    batch_size = 16
    support_images = torch.randn(5, 3, 224, 224)  # 5 support samples
    support_labels = torch.tensor([0, 0, 1, 1, 2])  # 3 classes
    query_images = torch.randn(10, 3, 224, 224)    # 10 query samples

    with torch.no_grad():
        results = model(support_images, support_labels, query_images)
        print(f"Query predictions shape: {results['probabilities'].shape}")
        print(f"Unknown samples detected: {results['unknown_mask'].sum().item()}")
