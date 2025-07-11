"""
%load_ext autoreload
%autoreload 2
"""
import torch
import os
import json
import mlflow
import mlflow.pytorch
from collections import defaultdict
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
from hcmus.utils import data_utils, viz_utils
from hcmus.utils import transform_utils
from hcmus.models.backbone import CLIPBackbone
from hcmus.models.prototype import PrototypicalNetwork
from hcmus.models.prototype import PrototypicalTrainer


def get_or_create_experiment():
    name = "/PrototypicalNetworks"
    try:
        mlflow.create_experiment(name)
    except:
        pass

    return mlflow.get_experiment_by_name(name).experiment_id


def create_fewshot_loader(datasets, split_name: str, n_way, n_shot, n_query, n_tasks = 100):
    sampler = TaskSampler(
        dataset=datasets[split_name],
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks
    )
    dataloader = DataLoader(
        datasets[split_name],
        batch_sampler=sampler,
        collate_fn=sampler.episodic_collate_fn
    )
    return dataloader


def setup_trainer():
    backbone = CLIPBackbone("ViT-B/32")
    model = PrototypicalNetwork(backbone)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = PrototypicalTrainer(model, optimizer, criterion)
    return trainer


def log_json_artifact(obj, filename: str):
    with open(filename, "w") as f:
        json.dump(obj, f)
    mlflow.log_artifact(filename, "json")
    os.remove(filename)


def log_torch_artifact(obj, filename: str):
    torch.save(obj, filename)
    mlflow.log_artifact(filename, "torch")
    os.remove(filename)


def convert_prototypes(prototypes, classes):
    result = {}
    for prototype, idx in zip(prototypes, classes):
        result[idx] = prototype
    return result


def merge_prototypes(prototypes: dict, converted: dict):
    for k, v in converted.items():
        if k not in prototypes:
            prototypes[k] = []

        prototypes[k].append(v)
    return prototypes


def export_prototypes(model, episode_prototypes, support_images, support_labels, classes):
    n_classes = len(classes)
    model.eval()
    support_features = model.encode_images(support_images)
    prototypes = model.compute_prototypes(support_features, support_labels, n_classes)
    converted_prototypes = convert_prototypes(prototypes, classes)
    episode_prototypes = merge_prototypes(episode_prototypes, converted_prototypes)
    return episode_prototypes


def choose_prototypes(model, prototypes: dict, datasets: dict, split_name: str):
    dataset = datasets[split_name]
    keys = sorted(prototypes.keys())
    best_prototypes = {}

    # Group samples by label more efficiently
    samples_by_label = defaultdict(list)
    for idx, (item) in enumerate(tqdm(dataset.samples, desc="Grouping samples...")):
        label = item["label"]
        samples_by_label[label].append(idx)

    # Process each class
    for k in tqdm(keys, desc="Computing distances..."):
        if k not in samples_by_label:
            # Handle case where class k has no samples
            best_prototypes[k] = prototypes[k][0]
            continue

        sample_indices = samples_by_label[k]
        prototype_tensors = torch.stack(prototypes[k])  # Shape: [num_prototypes, feature_dim]

        # Batch process images for this class
        batch_size = min(32, len(sample_indices))  # Adjust based on GPU memory
        total_distances = torch.zeros(len(prototypes[k]), device=prototype_tensors.device)

        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]

            # Load and encode batch of images
            batch_images = []
            for idx in batch_indices:
                image, _ = dataset[idx]
                batch_images.append(image)
            batch_images = torch.stack(batch_images)
            # Encode all images in the batch at once
            batch_features = model.encode_images(batch_images)  # Shape: [batch_size, feature_dim]

            # Compute distances between all batch features and all prototypes
            # batch_features: [batch_size, feature_dim]
            # prototype_tensors: [num_prototypes, feature_dim]
            distances = torch.cdist(batch_features, prototype_tensors)  # [batch_size, num_prototypes]

            # Sum distances for each prototype across the batch
            total_distances += distances.sum(dim=0)

        # Find prototype with minimum total distance
        best_idx = torch.argmin(total_distances).item()
        best_prototypes[k] = prototypes[k][best_idx]

    return best_prototypes


def evaluate_from_prototypes(model, prototypes, datasets, split_name: str):
    # prototypes = best_prototypes
    # model = trainer.model
    # split_name = "val"
    model.eval()
    prototypes_keys = list(prototypes.keys())
    prototypes_ts = torch.stack(list(prototypes.values()))
    loader = DataLoader(datasets[split_name], batch_size=16)
    accuracy = 0
    for images, labels in tqdm(loader, f"Evaluating {split_name}..."):
        emb = model.encode_images(images)
        dist = torch.cdist(emb, prototypes_ts)
        pred = torch.argmin(dist, dim=1)
        pred = [prototypes_keys[x.item()] for x in pred]
        torch.Tensor(pred)

        accuracy += sum(torch.tensor(pred) == labels)
    return accuracy / len(datasets[split_name])


def evaluate_from_sampler(model, datasets, split_name: str, n_shot: int):
    # prototypes = best_prototypes
    # model = trainer.model
    # split_name = "val"
    model.eval()
    support_split = "support"
    n_way = len(datasets[support_split].label2idx)
    fewshot_loader = create_fewshot_loader(datasets, support_split, n_way=n_way, n_shot=n_shot, n_query=0, n_tasks=1)
    support_images, support_labels, _, _, classes = next(iter(fewshot_loader))
    support_features = model.encode_images(support_images)
    prototypes = model.compute_prototypes(support_features, support_labels)

    loader = DataLoader(datasets[split_name], batch_size=16)
    accuracy = 0
    for images, labels in tqdm(loader, f"Evaluating {split_name}..."):
        emb = model.encode_images(images)
        dist = torch.cdist(emb, prototypes)
        pred = torch.argmin(dist, dim=1)
        pred = [classes[x.item()] for x in pred]
        torch.Tensor(pred)

        accuracy += sum(torch.tensor(pred) == labels)
    return accuracy / len(datasets[split_name])


def train_episode(
    datasets: dict,
    n_epoch: int = 32,
    n_way: int = 32,
    n_shot: int = 5,
    n_query: int = 5,
    n_tasks: int = 64
):

    """
    n_epoch: int = 32
    n_way: int = 32
    n_shot: int = 5
    n_query: int = 5
    n_tasks: int = 64
    """
    trainer = setup_trainer()
    experiment_id = get_or_create_experiment()
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params({
            "training_method": "episode",
            "n_epoch": n_epoch,
            "n_way": n_way,
            "n_shot": n_shot,
            "n_query": n_query,
            "n_tasks": n_tasks
        })

        log_json_artifact(datasets["train"].label2idx, "label2idx.json")
        log_json_artifact(datasets["train"].idx2label, "idx2label.json")

        train_dataloader = create_fewshot_loader(datasets, "train", n_way, n_shot, n_query, n_tasks)
        # desc = f"Train(n_way={n_way}, n_shot={n_shot})" + "={loss:.2f},{acc:.2f},{best:.2f},{val:.2f}"
        # bar = tqdm(range(n_epoch), desc=desc.format(loss=0, acc=0, best=0, val=0))
        best_accuracy = 0
        # episode_prototypes = {}

        for step in range(n_epoch):
            episode_loss = 0
            episode_acc = 0
            # episode = next(iter(train_dataloader))
            for episode in tqdm(train_dataloader, desc=f"Training epoch={step}..."):
                support_images, support_labels, query_images, query_labels, classes = episode

                # support_features = trainer.model.encode_images(support_images)
                # prototypes = trainer.model.compute_prototypes(support_features, support_labels)
                # query_features = trainer.model.encode_images(query_images)
                # dist = torch.cdist(query_features, prototypes)

                loss, acc = trainer.train_episode(support_images, support_labels, query_images, query_labels)

                # episode_prototypes = export_prototypes(trainer.model, episode_prototypes, support_images, support_labels, classes)
                # support_features = trainer.model.encode_images(support_images)
                # prototypes = trainer.model.compute_prototypes(support_features, support_labels, n_classes)
                # converted_prototypes = convert_prototypes(prototypes, classes)
                # episode_prototypes = merge_prototypes(episode_prototypes, converted_prototypes)
                episode_loss += loss
                episode_acc += acc

            mlflow.log_metrics({
                "train_loss": episode_loss / n_tasks,
                "train_accuracy": episode_acc / n_tasks
            }, step=step)

            val_accuracy = evaluate_from_sampler(trainer.model, datasets, "val", n_shot=n_shot)
            mlflow.log_metric("val_accuracy", val_accuracy, step=step)
            test_accuracy = evaluate_from_sampler(trainer.model, datasets, "test", n_shot=n_shot)
            mlflow.log_metric("test_accuracy", test_accuracy, step=step)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # best_prototypes = choose_prototypes(trainer.model, episode_prototypes, datasets, "train")
                # episode_prototypes = {k: [v] for k, v in best_prototypes.items()}
                # val_accuracy_prototype = evaluate_from_prototypes(trainer.model, best_prototypes, datasets, "val")
                mlflow.log_metric("best_accuracy", val_accuracy, step=step)
                # mlflow.log_metric("val_accuracy", val_accuracy_prototype, step=step)
                mlflow.pytorch.log_model(trainer.model, f"checkpoint/model_{step}")
                mlflow.pytorch.log_model(trainer.model, f"model")
                # log_torch_artifact(best_prototypes, f"prototypes_{step}.pt")
                # log_torch_artifact(best_prototypes, f"prototypes.pt")

if __name__ == "__main__":
    splits = data_utils.get_data_splits()
    transform_train, transform_test = transform_utils.get_transforms_downscale_random_v2()
    datasets = data_utils.get_image_datasets(splits, transform_test, transform_test, random_margin=0.0)

    # train_episode(datasets, n_way=64, n_shot=5)
    # train_episode(datasets, n_way=64, n_shot=5, n_tasks=16)
    # train_episode(datasets, n_way=64, n_shot=7, n_tasks=16)
    # train_episode(datasets, n_way=99, n_shot=5, n_tasks=16)
    # train_episode(datasets, n_way=99, n_shot=7, n_tasks=16)
    train_episode(datasets, n_way=99, n_shot=11, n_tasks=16, n_epoch=64)



