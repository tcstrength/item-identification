import torch
from typing import List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hcmus.lbs import LabelStudioConnector
from hcmus.data._data_augmentation import DataAugmentation


class LbsDataset(Dataset):
    def __init__(
        self,
        connector: LabelStudioConnector,
        device: str,
        augmentation: bool = False
    ):
        self._dataset = []
        self._label = []
        self._to_tensor = T.ToTensor()
        self._to_image = T.ToPILImage()
        self._connector = connector
        self._device = device

        if augmentation:
            self._augment = DataAugmentation(device=device, img_size=(512, 512))
        else:
            self._augment = None

        dataset, labels = self.__download_dataset()
        logger.info(f"Number of labels: {len(labels)}")
        logger.info(f"Number of data points: {len(dataset)}")
        self._dataset = dataset
        self._labels = labels

    def __download_image(self, task):
        image = self._connector.get_image(task)
        return {
            "image": image,
            "task": task,
        }

    def __download_dataset(self) -> List:
        tasks = self._connector.get_tasks(1, 100, page_size=512)
        dataset = []
        labels = {
            "@background": 0
        }

        to_download = []
        for task in tasks:
            if task.is_labeled == 0: continue
            to_download.append(task)

        for task in to_download:
            for ann in task.annotations:
                tmp = ann.result[0].value.rectanglelabels
                if len(tmp) > 1:
                    logger.warning(f"Unexpected labels: {tmp}")
                tmp = tmp[0]
                if tmp not in labels:
                    labels[tmp] = len(labels)

        # for task in tqdm(to_download, "Downloading images"):
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.__download_image, task): task for task in to_download}

            for future in tqdm(as_completed(futures), total=len(to_download), desc="Downloading images"):
                dataset.append(future.result())

        return dataset, labels

    def __getitem__(self, idx):
        item = self._dataset[idx]
        path = item.get("image")
        task = item.get("task")
        image = Image.open(path).convert("RGB")
        boxes = []
        labels = []
        width, height = image.size

        for ann in task.annotations:
            ann = task.annotations[0]
            rect = ann.result[0].value
            label = rect.rectanglelabels[0]
            label_idx = self._labels[label]
            x_min = width * (rect.x / 100)
            y_min = height * (rect.y / 100)
            x_max = x_min + (width * (rect.width / 100))
            y_max = y_min + (height * (rect.height / 100))
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label_idx)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=self._device),
            "labels": torch.tensor(labels, dtype=torch.int64, device=self._device)
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
