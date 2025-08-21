import time
from huggingface_hub import save_torch_model
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from hcmus.models.backbone import BaseBackbone


class TransferNetwork(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        output_dim: int,
        dropout: float = 0.2,
        feature_dim: int = 512,
    ):
        super().__init__()

        self._backbone = backbone
        self._feature_dim = feature_dim
        self._classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._backbone.output_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim)
        )


    def forward(self, x):
        features = self._backbone(x)
        logits = self._classifier(features)
        return logits

class TransferTrainer:
    def __init__(self, model, device=None, learning_rate=0.001, optimizer_type='adam',
                 mlflow_experiment_name=None, mlflow_run_name=None):
        """
        Initialize the TransferTrainer

        Args:
            model: PyTorch model that outputs logits (not softmax)
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            mlflow_experiment_name: Name of MLflow experiment
            mlflow_run_name: Name of MLflow run
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function (CrossEntropyLoss expects logits)
        self.criterion = nn.CrossEntropyLoss()

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        # Optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # MLflow setup
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_run = None

        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)

    def train(self, n_epochs, train_loader, val_loader=None, test_loader=None, verbose=True):
        """
        Train the model for n_epochs

        Args:
            n_epochs: Number of epochs to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            verbose: Whether to print training progress

        Returns:
            dict: Training history containing losses and validation metrics
        """
        # Start MLflow run
        if self.mlflow_experiment_name:
            self.mlflow_run = mlflow.start_run(run_name=self.mlflow_run_name)

            # Log hyperparameters
            mlflow.log_params({
                'learning_rate': self.learning_rate,
                'optimizer_type': self.optimizer_type,
                'n_epochs': n_epochs,
                'device': str(self.device),
                'model_name': self.model.__class__.__name__
            })

            # Count total parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            mlflow.log_params({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })

        self.model.train()

        best_val_accuracy = 0

        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Training phase
            train_loss = 0.0
            train_batches = 0

            if verbose:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            else:
                pbar = train_loader

            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                train_loss += loss.item()
                train_batches += 1

                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Calculate average training loss
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)

            # Log training loss to MLflow
            if self.mlflow_run:
                mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, verbose=False)
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                new_best_model = val_metrics["accuracy"] > best_val_accuracy
                if new_best_model:
                    best_val_accuracy = val_metrics["accuracy"]
                    self.save_model("best_model.pt")

                # Log validation metrics to MLflow
                if self.mlflow_run:
                    if new_best_model:
                        mlflow.pytorch.log_model(self.model, "best_model")

                    mlflow.log_metric('val_loss', val_metrics['loss'], step=epoch)
                    mlflow.log_metric('val_accuracy', val_metrics['accuracy'], step=epoch)
                    mlflow.log_metric('val_f1_macro', val_metrics['f1_macro'], step=epoch)
                    mlflow.log_metric('val_f1_micro', val_metrics['f1_micro'], step=epoch)


            # Print epoch summary
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch {epoch+1}/{n_epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                if val_loader is not None:
                    print(f'  Val Loss: {val_metrics["loss"]:.4f}')
                    print(f'  Val Accuracy: {val_metrics["accuracy"]:.4f}')
                print(f'  Time: {epoch_time:.2f}s')
                print('-' * 50)

        # Log final model to MLflow
        if self.mlflow_run:
            mlflow.pytorch.log_model(self.model, "model")
            if test_loader is not None:
                test_metrics = self.evaluate(test_loader, verbose=False)
                mlflow.log_metric('test_loss', test_metrics['loss'], step=epoch)
                mlflow.log_metric('test_accuracy', test_metrics['accuracy'], step=epoch)
                mlflow.log_metric('test_f1_macro', test_metrics['f1_macro'], step=epoch)
                mlflow.log_metric('test_f1_micro', test_metrics['f1_micro'], step=epoch)

                if verbose:
                    print(f'  Test Loss: {test_metrics["loss"]:.4f}')
                    print(f'  Test Accuracy: {test_metrics["accuracy"]:.4f}')

        self.end_run()
        # return {
        #     'train_losses': self.train_losses,
        #     'val_losses': self.val_losses,
        #     'val_accuracies': self.val_accuracies
        # }

    def evaluate(self, test_loader, verbose=True, log_mlflow: bool=False):
        """
        Evaluate the model on test data

        Args:
            test_loader: Test data loader
            verbose: Whether to print evaluation results

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print("Start evaluating...")
        self.model.eval()

        all_predictions = []
        all_labels = []
        test_loss = 0.0
        test_batches = 0

        with torch.no_grad():
            if verbose:
                pbar = tqdm(test_loader, desc='Evaluating')
            else:
                pbar = test_loader

            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                # Get predictions (convert logits to class predictions)
                predictions = torch.argmax(logits, dim=1)

                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Statistics
                test_loss += loss.item()
                test_batches += 1

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate all metrics
        accuracy = accuracy_score(all_labels, all_predictions)

        # Micro averages
        precision_micro = precision_score(all_labels, all_predictions, average='micro')
        recall_micro = recall_score(all_labels, all_predictions, average='micro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')

        # Macro averages
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')

        # Average test loss
        avg_test_loss = test_loss / test_batches

        metrics = {
            'loss': avg_test_loss,
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }

        # Print results
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Loss: {avg_test_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nMicro Averages:")
            print(f"  Precision: {precision_micro:.4f}")
            print(f"  Recall: {recall_micro:.4f}")
            print(f"  F1-Score: {f1_micro:.4f}")
            print("\nMacro Averages:")
            print(f"  Precision: {precision_macro:.4f}")
            print(f"  Recall: {recall_macro:.4f}")
            print(f"  F1-Score: {f1_macro:.4f}")
            print("="*50)

        # Log final test metrics to MLflow
        if self.mlflow_run and log_mlflow:
            mlflow.log_metrics({
                'test_loss': avg_test_loss,
                'test_accuracy': accuracy,
                'test_precision_micro': precision_micro,
                'test_recall_micro': recall_micro,
                'test_f1_micro': f1_micro,
                'test_precision_macro': precision_macro,
                'test_recall_macro': recall_macro,
                'test_f1_macro': f1_macro
            })

        return metrics

    def save_model(self, filepath):
        """Save the model state dict"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the model state dict"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {filepath}")

    def end_run(self):
        """End the current MLflow run"""
        if self.mlflow_run:
            mlflow.end_run()
            self.mlflow_run = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures MLflow run is ended"""
        self.end_run()

if __name__ == "__main__":
    from hcmus.utils import data_utils
    from hcmus.utils import transform_utils
    from hcmus.models.backbone import CLIPBackbone

    splits = data_utils.get_data_splits()
    transforms_train, transforms_test = transform_utils.get_transforms_downscale_transfer_learning()
    datasets = data_utils.get_image_datasets_v2(splits, transform_train=transforms_train, transform_test=transforms_test)
    dataloaders = data_utils.get_data_loaders_v2(datasets, {
        "train": True,
        "val": False,
        "test": False
    })

    backbone = CLIPBackbone("ViT-B/32")
    model = TransferNetwork(backbone, 99)

    trainer = TransferTrainer(model)
    trainer.train(1, dataloaders["train"], dataloaders["val"])
    trainer.evaluate(dataloaders["test"])

    image, label, _ = train_dataset[0]

    model(image)
