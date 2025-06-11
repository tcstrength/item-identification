import torch
from typing import Dict, List
from loguru import logger
from torchvision import transforms as T
from PIL import Image, ImageOps
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


class CroppedObjectClassificationDataset(Dataset):
    def __init__(self, data_list: List, label2idx: Dict=None, transforms=None, skip_labels=[]):
        """
        data_list: List of dicts with 'image' path and 'target' with 'boxes' and 'labels'.
        transforms: Optional transforms applied to each cropped object.
        """
        self.samples = []
        self.transforms = transforms
        self.label2idx = label2idx

        valid_labels = []
        for entry in data_list:
            labels = entry['target']['labels']
            labels = [x for x in labels if x not in skip_labels]
            valid_labels.extend(labels)

        if label2idx is None:
            self.label2idx = {v: k for k, v in enumerate(set(valid_labels))}
            logger.info(f"Auto infer `label2idx` mapping, mapping length: {len(self.label2idx)}.")

        self.idx2label = {v: k for v, k in self.label2idx.items()}

        for entry in data_list:
            img_path = entry['image']
            boxes = entry['target']['boxes']
            labels = entry['target']['labels']

            for box, label in zip(boxes, labels):
                if label in skip_labels:
                    continue

                labels.append(label)
                self.samples.append({
                    'image': img_path,
                    'box': box,
                    'label': self.label2idx[label]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image']
        box = sample['box']
        label = sample['label']

        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # Crop image based on box [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))

        if self.transforms:
            cropped = self.transforms(cropped)

        return cropped, label
