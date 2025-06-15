import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import torchvision.transforms as transforms
import mlflow
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


AVAILABLE_BACKBONES = {
    'lightweight': [
        'mobilenetv3_large_100',
        'efficientnet_b0',
        'resnet18'
    ],
    'balanced': [
        'resnet50',
        'efficientnet_b3',
        'regnety_032',
        'convnext_tiny'
    ],
    'high_performance': [
        'resnet101',
        'efficientnet_b5',
        'convnext_base',
        'swin_base_patch4_window7_224',
        'vit_base_patch16_224'
    ],
    'state_of_the_art': [
        'convnext_large',
        'swin_large_patch4_window7_224',
        'vit_large_patch16_224',
        'eva_large_patch14_196'
    ]
}


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning with open-world capability.
    """
    def __init__(self, backbone_name: str = 'resnet50', embedding_dim: int = 512,
                 dropout_rate: float = 0.1):
        super().__init__()

        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )

        # Get backbone output dimension
        backbone_dim = self.backbone.num_features

        # Projection head for embedding
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward(self, x):
        """Forward pass to get embeddings."""
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, dim=1)  # L2 normalize


class OpenWorldFewShotClassifier(nn.Module):
    """
    Open-world few-shot classifier that can:
    1. Classify known classes using prototypes
    2. Detect unknown/novel classes
    3. Adapt to new classes with few examples
    """

    def __init__(self, backbone_name: str = 'resnet50', embedding_dim: int = 512,
                 temperature: float = 10.0, unknown_threshold: float = 0.5):
        super().__init__()

        self.feature_extractor = PrototypicalNetwork(backbone_name, embedding_dim)
        self.temperature = temperature
        self.unknown_threshold = unknown_threshold
        self.prototypes = {}  # Store class prototypes
        self.class_names = []
        self.embedding_dim = embedding_dim

        # Outlier detection parameters
        self.outlier_detector = None
        self.fit_outlier_detector = True

    def compute_prototypes(self, support_embeddings: torch.Tensor,
                          support_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute prototypes for each class in support set."""
        prototypes = {}
        unique_labels = torch.unique(support_labels)

        for label in unique_labels:
            label_mask = support_labels == label
            class_embeddings = support_embeddings[label_mask]
            # Prototype is the mean of support embeddings for this class
            prototypes[label.item()] = torch.mean(class_embeddings, dim=0)

        return prototypes

    def compute_distances(self, query_embeddings: torch.Tensor,
                         prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute distances between queries and prototypes."""
        n_queries = query_embeddings.size(0)
        n_classes = len(prototypes)

        distances = torch.zeros(n_queries, n_classes, device=query_embeddings.device)

        for i, (class_id, prototype) in enumerate(prototypes.items()):
            # Euclidean distance
            distances[:, i] = torch.norm(query_embeddings - prototype.unsqueeze(0), dim=1)

        return distances

    def detect_unknown_classes(self, query_embeddings: torch.Tensor,
                              prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Detect unknown/novel classes using distance-based thresholding."""
        if not prototypes:
            return torch.ones(query_embeddings.size(0), dtype=torch.bool)

        distances = self.compute_distances(query_embeddings, prototypes)
        min_distances = torch.min(distances, dim=1)[0]

        # Points with distance > threshold are considered unknown
        unknown_mask = min_distances > self.unknown_threshold

        return unknown_mask

    def cluster_unknown_samples(self, unknown_embeddings: torch.Tensor,
                              eps: float = 0.3, min_samples: int = 2) -> np.ndarray:
        """Cluster unknown samples to discover potential new classes."""
        if len(unknown_embeddings) < min_samples:
            return np.array([-1] * len(unknown_embeddings))

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(unknown_embeddings.cpu().numpy())

        return cluster_labels

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor,
                query_images: torch.Tensor, return_embeddings: bool = False):
        """
        Forward pass for few-shot classification.

        Args:
            support_images: Support set images [N_support, C, H, W]
            support_labels: Support set labels [N_support]
            query_images: Query set images [N_query, C, H, W]
            return_embeddings: Whether to return embeddings
        """
        # Extract embeddings
        support_embeddings = self.feature_extractor(support_images)
        query_embeddings = self.feature_extractor(query_images)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)

        # Detect unknown classes
        unknown_mask = self.detect_unknown_classes(query_embeddings, prototypes)

        # Compute distances and logits for known classes
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances * self.temperature

        # Handle unknown samples
        unknown_embeddings = query_embeddings[unknown_mask]
        novel_clusters = None
        if len(unknown_embeddings) > 0:
            novel_clusters = self.cluster_unknown_samples(unknown_embeddings)

        results = {
            'logits': logits,
            'unknown_mask': unknown_mask,
            'novel_clusters': novel_clusters,
            'prototypes': prototypes
        }

        if return_embeddings:
            results['query_embeddings'] = query_embeddings
            results['support_embeddings'] = support_embeddings

        return results

    def predict(self, support_images: torch.Tensor, support_labels: torch.Tensor,
                query_images: torch.Tensor) -> Dict:
        """Make predictions on query images."""
        self.eval()
        with torch.no_grad():
            results = self.forward(support_images, support_labels, query_images)

            # Get predictions for known classes
            probabilities = F.softmax(results['logits'], dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

            # Mark unknown samples
            predicted_classes[results['unknown_mask']] = -1  # -1 for unknown

            return {
                'predictions': predicted_classes,
                'probabilities': probabilities,
                'unknown_mask': results['unknown_mask'],
                'novel_clusters': results['novel_clusters']
            }


class OpenWorldTrainer:
    """Trainer for open-world few-shot learning."""

    def __init__(self, model: OpenWorldFewShotClassifier, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def episodic_train_step(self, support_images: torch.Tensor, support_labels: torch.Tensor,
                           query_images: torch.Tensor, query_labels: torch.Tensor):
        """Single episodic training step."""
        self.model.train()

        results = self.model(support_images, support_labels, query_images)

        # Compute loss only for known classes (not unknown)
        known_mask = ~results['unknown_mask']
        if known_mask.sum() > 0:
            known_logits = results['logits'][known_mask]
            known_labels = query_labels[known_mask]
            loss = F.cross_entropy(known_logits, known_labels)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss, results

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   n_way: int = 5, n_shot: int = 5, n_query: int = 15):
        """Train for one epoch using episodic training."""
        total_loss = 0.0
        total_acc = 0.0
        num_episodes = 0

        for batch_idx, batch in enumerate(dataloader):
            # Create episodes from batch
            episode = self.create_episode(batch, n_way, n_shot, n_query)
            if episode is None:
                continue

            support_images, support_labels, query_images, query_labels = episode

            optimizer.zero_grad()
            loss, results = self.episodic_train_step(
                support_images, support_labels, query_images, query_labels
            )

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            # Calculate accuracy for known samples
            known_mask = ~results['unknown_mask']
            if known_mask.sum() > 0:
                predictions = torch.argmax(results['logits'][known_mask], dim=1)
                accuracy = (predictions == query_labels[known_mask]).float().mean()
                total_acc += accuracy.item()

            total_loss += loss.item()
            num_episodes += 1

        return total_loss / num_episodes, total_acc / num_episodes

    def create_episode(self, batch, n_way: int, n_shot: int, n_query: int):
        """Create an episode from a batch of data."""
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        unique_labels = torch.unique(labels)

        if len(unique_labels) < n_way:
            return None

        # Sample n_way classes
        selected_classes = unique_labels[torch.randperm(len(unique_labels))[:n_way]]

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for i, class_label in enumerate(selected_classes):
            class_mask = labels == class_label
            class_images = images[class_mask]

            if len(class_images) < n_shot + n_query:
                continue

            # Randomly sample support and query
            indices = torch.randperm(len(class_images))

            support_indices = indices[:n_shot]
            query_indices = indices[n_shot:n_shot + n_query]

            support_images.append(class_images[support_indices])
            support_labels.extend([i] * n_shot)  # Use episode-specific labels

            query_images.append(class_images[query_indices])
            query_labels.extend([i] * n_query)

        if len(support_images) == 0:
            return None

        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.tensor(support_labels, device=self.device)
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.tensor(query_labels, device=self.device)

        return support_images, support_labels, query_images, query_labels


class OpenWorldTrainerEvaluator:
    """Complete trainer and evaluator for open-world few-shot learning."""

    def __init__(self, model, device, experiment_id: int):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}

    def train_episode(self, train_dataset, n_way=5, n_shot=5, n_query=10):
        """Train on a single episode."""
        self.model.train()

        # Create episode from training data (only known classes)
        support_images, support_labels, query_images, query_labels = \
            train_dataset.create_few_shot_episode(n_way, n_shot, n_query, include_unknown=False)

        # Convert to tensors
        support_images = torch.stack([train_dataset.transform(img) for img in support_images]).to(self.device)
        support_labels = torch.tensor(support_labels).to(self.device)
        query_images = torch.stack([train_dataset.transform(img) for img in query_images]).to(self.device)
        query_labels = torch.tensor(query_labels).to(self.device)

        # Forward pass
        results = self.model(support_images, support_labels, query_images)

        # Compute loss (only for known classes)
        loss = F.cross_entropy(results['logits'], query_labels)

        # Compute accuracy
        predictions = torch.argmax(results['logits'], dim=1)
        accuracy = (predictions == query_labels).float().mean().item()

        return loss, accuracy, len(query_labels)

    def train_epoch(self, train_dataset, optimizer, n_episodes=100,
                   n_way=5, n_shot=5, n_query=10):
        """Train for one epoch."""
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        pbar = tqdm(range(n_episodes), desc="Training Episodes")

        for episode in pbar:
            optimizer.zero_grad()

            try:
                loss, accuracy, n_samples = self.train_episode(
                    train_dataset, n_way, n_shot, n_query
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy * n_samples
                total_samples += n_samples

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })

            except Exception as e:
                print(f"Episode {episode} failed: {e}")
                continue

        avg_loss = total_loss / n_episodes
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0

        return avg_loss, avg_accuracy

    def evaluate_open_world(self, test_dataset, n_way=5, n_shot=5, n_query=10,
                           n_episodes=100, adjust_threshold=True):
        """Evaluate on test dataset with unknown classes."""
        self.model.eval()

        all_predictions = []
        all_true_labels = []
        all_confidences = []

        print("Evaluating open-world performance...")

        with torch.no_grad():
            for episode in tqdm(range(n_episodes), desc="Test Episodes"):
                try:
                    # Create episode with unknown samples
                    support_images, support_labels, query_images, query_labels = \
                        test_dataset.create_few_shot_episode(
                            n_way, n_shot, n_query, include_unknown=True
                        )

                    # Convert to tensors
                    support_images = torch.stack([test_dataset.transform(img) for img in support_images]).to(self.device)
                    support_labels = torch.tensor(support_labels).to(self.device)
                    query_images = torch.stack([test_dataset.transform(img) for img in query_images]).to(self.device)
                    query_labels = torch.tensor(query_labels).to(self.device)

                    # Get predictions
                    results = self.model.predict(support_images, support_labels, query_images)

                    predictions = results['predictions'].cpu().numpy()
                    probabilities = results['probabilities'].cpu().numpy()
                    unknown_mask = results['unknown_mask'].cpu().numpy()

                    # Convert episode labels back to predictions
                    final_predictions = []
                    confidences = []

                    for i, (pred, prob, is_unknown) in enumerate(zip(predictions, probabilities, unknown_mask)):
                        if is_unknown or pred == -1:
                            final_predictions.append(-1)  # Unknown
                            confidences.append(0.0)
                        else:
                            final_predictions.append(pred)
                            confidences.append(np.max(prob))

                    all_predictions.extend(final_predictions)
                    all_true_labels.extend(query_labels.cpu().numpy())
                    all_confidences.extend(confidences)

                except Exception as e:
                    print(f"Evaluation episode {episode} failed: {e}")
                    continue

        # Compute metrics
        metrics = self.compute_open_world_metrics(
            all_true_labels, all_predictions, all_confidences
        )

        return metrics

    def compute_open_world_metrics(self, y_true, y_pred, confidences):
        """Compute comprehensive open-world metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confidences = np.array(confidences)

        # Separate known and unknown samples
        known_mask = y_true != -1
        unknown_mask = y_true == -1

        metrics = {}

        # Overall accuracy
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

        # Known class accuracy (excluding unknown samples)
        if known_mask.sum() > 0:
            known_true = y_true[known_mask]
            known_pred = y_pred[known_mask]
            metrics['known_accuracy'] = accuracy_score(known_true, known_pred)
        else:
            metrics['known_accuracy'] = 0.0

        # Unknown detection metrics
        if unknown_mask.sum() > 0:
            # True positives: unknown samples correctly identified as unknown
            unknown_detected = (y_pred[unknown_mask] == -1).sum()
            metrics['unknown_recall'] = unknown_detected / unknown_mask.sum()

            # False positives: known samples incorrectly identified as unknown
            total_predicted_unknown = (y_pred == -1).sum()

            if total_predicted_unknown > 0:
                metrics['unknown_precision'] = unknown_detected / total_predicted_unknown
            else:
                metrics['unknown_precision'] = 0.0

            # F1 score for unknown detection
            if metrics['unknown_precision'] + metrics['unknown_recall'] > 0:
                metrics['unknown_f1'] = 2 * (metrics['unknown_precision'] * metrics['unknown_recall']) / \
                                       (metrics['unknown_precision'] + metrics['unknown_recall'])
            else:
                metrics['unknown_f1'] = 0.0
        else:
            metrics['unknown_recall'] = 0.0
            metrics['unknown_precision'] = 0.0
            metrics['unknown_f1'] = 0.0

        # H-measure (Harmonic mean of known accuracy and unknown recall)
        if metrics['known_accuracy'] + metrics['unknown_recall'] > 0:
            metrics['h_measure'] = 2 * metrics['known_accuracy'] * metrics['unknown_recall'] / \
                                  (metrics['known_accuracy'] + metrics['unknown_recall'])
        else:
            metrics['h_measure'] = 0.0

        # Confidence statistics
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)

        return metrics

    def train_full(self, train_dataset, test_dataset, optimizer, scheduler=None,
                   n_epochs=50, n_episodes_per_epoch=100, n_way=5, n_shot=5, n_query=10,
                   eval_every=5, save_best=True):
        """Complete training loop with evaluation."""

        best_h_measure = 0.0

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Episodes per epoch: {n_episodes_per_epoch}")
        print(f"Few-shot setup: {n_way}-way {n_shot}-shot with {n_query} queries")

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")

            # Training
            train_loss, train_acc = self.train_epoch(
                train_dataset, optimizer, n_episodes_per_epoch, n_way, n_shot, n_query
            )

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc
            })

            # Evaluation
            if (epoch + 1) % eval_every == 0:
                eval_metrics = self.evaluate_open_world(
                    test_dataset, n_way, n_shot, n_query, n_episodes=50
                )

                print(f"Evaluation Results:")
                print(f"  Overall Accuracy: {eval_metrics['overall_accuracy']:.4f}")
                print(f"  Known Accuracy: {eval_metrics['known_accuracy']:.4f}")
                print(f"  Unknown Recall: {eval_metrics['unknown_recall']:.4f}")
                print(f"  Unknown Precision: {eval_metrics['unknown_precision']:.4f}")
                print(f"  Unknown F1: {eval_metrics['unknown_f1']:.4f}")
                print(f"  H-Measure: {eval_metrics['h_measure']:.4f}")

                self.history['val_metrics'].append(eval_metrics)

                mlflow.log_metrics(eval_metrics, step=epoch)
                # Save best model
                if save_best and eval_metrics['h_measure'] > best_h_measure:
                    best_h_measure = eval_metrics['h_measure']
                    mlflow.pytorch.log_model(self.model, "model")
                    print(f"New best model saved! H-measure: {best_h_measure:.4f}")

            # Update learning rate
            if scheduler:
                scheduler.step()

        return self.history


def get_transforms(image_size: int = 224):
    """Get data transforms for training and testing."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


# Example usage
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print recommended backbones
    print("Recommended Backbones:")
    for category, backbones in AVAILABLE_BACKBONES.items():
        print(f"\n{category.upper()}:")
        for backbone in backbones:
            print(f"  - {backbone}")

    # Initialize model with different backbones
    print(f"\nInitializing models on {device}...")

    # Lightweight model
    lightweight_model = OpenWorldFewShotClassifier(
        backbone_name='mobilenetv3_large_100',
        embedding_dim=256,
        temperature=10.0,
        unknown_threshold=0.6
    )

    # Balanced model
    balanced_model = OpenWorldFewShotClassifier(
        backbone_name='resnet50',
        embedding_dim=512,
        temperature=10.0,
        unknown_threshold=0.5
    )

    # High performance model
    high_perf_model = OpenWorldFewShotClassifier(
        backbone_name='convnext_base',
        embedding_dim=768,
        temperature=12.0,
        unknown_threshold=0.4
    )

    print("Models initialized successfully!")

    # Example of model parameters
    print(f"\nModel Parameters:")
    print(f"Lightweight: {sum(p.numel() for p in lightweight_model.parameters())/1e6:.1f}M")
    print(f"Balanced: {sum(p.numel() for p in balanced_model.parameters())/1e6:.1f}M")
    print(f"High Performance: {sum(p.numel() for p in high_perf_model.parameters())/1e6:.1f}M")

    # Example training setup
    model = balanced_model
    trainer = OpenWorldTrainer(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\nTrainer setup complete!")
    print("Use trainer.train_epoch(dataloader, optimizer) to train the model.")
