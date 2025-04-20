import torch
from typing import List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hcmus.lbs import LabelStudioConnector

class TorchDataset(Dataset):
    def __init__(
        self,
        connector: LabelStudioConnector,
        device: str = "cpu"
    ):
        self._dataset = []
        self._label = []
        self._to_tensor = T.ToTensor()
        self._to_image = T.ToPILImage()
        self._connector = connector
        self._device = device
        # Reserved for augmentations
        self._augment = None

        tasks = self._connector.get_tasks(1, 100)
        self._labels = self._connector.extract_labels(tasks)
        self._dataset = self._connector.download_dataset(tasks, self._labels)

        logger.info(f"Number of labels: {len(self._labels)}")
        logger.info(f"Number of data points: {len(self._dataset)}")

    def __getitem__(self, idx):
        item = self._dataset[idx]
        path = item.get("image")
        target = item.get("target")
        image = Image.open(path).convert("RGB")

        target = {
            "boxes": torch.tensor(target.get("boxes"), dtype=torch.float32, device=self._device),
            "labels": torch.tensor(target.get("labels"), dtype=torch.int64, device=self._device)
        }

        if self._augment:
            image, target = self._augment(image, target)

        tensor = self._to_tensor(image).to(self._device)
        return tensor, target

    def __len__(self):
        return len(self._dataset)

    def tensor_to_image(self, tensor: torch.Tensor) -> Image:
        return self._to_image(tensor)

    def get_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        pin_memory: bool = False
    ) -> DataLoader:
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=lambda x: tuple(zip(*x))
        )
        return dataloader
