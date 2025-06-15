import torch
import timm
import torch.nn as nn
import mlflow
from torch.nn import functional as F
from loguru import logger
from tqdm import tqdm

class ModelTrainer:
    """
    A comprehensive trainer class for both regular classification and ArcFace models.
    Handles training, evaluation, and model building with frozen backbones.
    """

    def __init__(self, device, num_classes, train_loader=None, val_loader=None):
        """
        Initialize the ModelTrainer.

        Args:
            device: torch.device for computations
            num_classes: number of classes for classification
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        self.device = device
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_data_loaders(self, train_loader, val_loader):
        """Set or update the data loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader


    def train_epoch(self, model, criterion, optimizer):
        """
        Train the model for one epoch.

        Args:
            model: the model to train
            criterion: loss function
            optimizer: optimizer for training

        Returns:
            tuple: (average_loss, accuracy)
        """
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            if self._is_arcface_model(model):
                # For ArcFace model, get embeddings and use ArcFace loss
                embeddings = model(images)
                loss = criterion(embeddings, labels)

                # For accuracy calculation, compute cosine similarity
                with torch.no_grad():
                    embeddings_norm = F.normalize(embeddings, dim=1)
                    W_norm = F.normalize(criterion.W, dim=0)
                    logits = torch.matmul(embeddings_norm, W_norm) * criterion.s
                    _, predicted = logits.max(1)
            else:
                # For regular classification model
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self, model, criterion=None):
        """
        Evaluate the model on validation data.

        Args:
            model: the model to evaluate
            criterion: loss function (needed for ArcFace models)

        Returns:
            float: accuracy percentage
        """
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                if self._is_arcface_model(model):
                    embeddings = model(images)

                    if criterion is not None:
                        # Use criterion weights for evaluation
                        W_norm = F.normalize(criterion.W, dim=0)
                        logits = torch.matmul(embeddings, W_norm) * criterion.s
                        _, predicted = logits.max(1)
                    else:
                        # Fallback: use or create evaluation classifier
                        predicted = self._get_arcface_predictions(model, embeddings)
                else:
                    # For regular classification model
                    outputs = model(images)
                    _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def train_model(self, model, criterion, optimizer, num_epochs, scheduler=None):
        """
        Train the model for multiple epochs.

        Args:
            model: the model to train
            criterion: loss function
            optimizer: optimizer for training
            num_epochs: number of training epochs
            scheduler: learning rate scheduler (optional)

        Returns:
            dict: training history with losses and accuracies
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Training phase
            train_loss, train_acc = self.train_epoch(model, criterion, optimizer)

            # Validation phase
            val_acc = self.evaluate(model, criterion)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")

            # Log progress
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

        return history

    def run(self, experiment_id, model_name, emb_size, lr: float=1e-3, epochs: int = 128):
        with mlflow.start_run(experiment_id=experiment_id, run_name="arc-" + model_name):
            logger.info(f"\n🔍 Training with frozen backbone: {model_name}")

            model = ArcModel(model_name, embedding_size=emb_size)
            criterion = ArcFaceLoss(embedding_size=emb_size, num_classes=self.num_classes)

            # Only train the new layers (embedding layer and batch norm)
            trainable_params = []
            for name, param in model.named_parameters():
                if 'backbone' not in name:  # Don't train backbone
                    trainable_params.append(param)
            trainable_params.extend(criterion.parameters())  # Include ArcFace weights

            optimizer = torch.optim.Adam(trainable_params, lr=lr)

            best_acc = 0
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                train_loss, train_acc = self.train_model(model, criterion, optimizer)
                val_acc = self.evaluate_model_with_criterion(model, criterion)

                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
                mlflow.log_metrics({
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc
                }, step=epoch)

                if val_acc > best_acc:
                    best_acc = val_acc
                    mlflow.log_metric({
                        "val_best_accuracy": best_acc
                    })
                    mlflow.pytorch.log_model(model, "model")

    def _is_arcface_model(self, model):
        """Check if the model is an ArcFace model."""
        # This assumes ArcModel is defined elsewhere in your codebase
        return hasattr(model, '__class__') and model.__class__.__name__ == 'ArcModel'

    def _get_arcface_predictions(self, model, embeddings):
        """Get predictions for ArcFace model without criterion."""
        # Check if model has classifier weights
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'W'):
            W_norm = F.normalize(model.classifier.W, dim=0)
            logits = torch.matmul(embeddings, W_norm)
            _, predicted = logits.max(1)
        else:
            # Create evaluation classifier if it doesn't exist
            if not hasattr(model, 'eval_classifier'):
                model.eval_classifier = nn.Linear(
                    embeddings.size(1),
                    self.num_classes
                ).to(self.device)
            logits = model.eval_classifier(embeddings)
            _, predicted = logits.max(1)

        return predicted

class ArcModel(nn.Module):
    def __init__(self, backbone_name, embedding_size=512):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        self.freeze_backbone(self.backbone)
        self.embedding = nn.Linear(self.backbone.num_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        # Add classification weights for evaluation
        self.classifier = None

    def freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.bn(x)
        return F.normalize(x, dim=1)  # Return normalized embeddings

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        # Embeddings should already be normalized from model
        W = F.normalize(self.W, dim=0)
        cosine = torch.matmul(embeddings, W)  # [B, C]

        # Clamp cosine values to prevent numerical issues
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        # Create one-hot encoding
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(embeddings.device)

        # Apply margin only to target class
        logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)

        return F.cross_entropy(logits, labels)
