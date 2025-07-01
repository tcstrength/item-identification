import random
import numpy as np
import torch
from typing import Callable, Dict, List
from loguru import logger
from torchvision import transforms as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from easyfsl.datasets import FewShotDataset
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


def _crop_image(img_path, box, transforms, random_margin: float = 0.2):
    box = map(int, box)
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    max_w, max_h = image.size

    # Crop image based on box [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    margin_w = (x2 - x1) * (random.random() - 0.5) * random_margin / 2
    margin_h = (y2 - y1) * (random.random() - 0.5) * random_margin / 2
    x1 = max(x1 - margin_w, 0)
    y1 = max(y1 - margin_h, 0)
    x2 = min(x2 + margin_w, max_w)
    y2 = min(y2 + margin_w, max_h)
    # print(
    #     max(x1 - margin_w, 0),
    #     max(y1 - margin_h, 0),
    #     min(x2 + margin_w, max_w),
    #     min(y2 + margin_h, max_h)
    # )
    cropped = image.crop((
        max(x1 - margin_w, 0),
        max(y1 - margin_h, 0),
        min(x2 + margin_w, max_w),
        min(y2 + margin_h, max_h)
    ))

    if transforms:
        cropped = transforms(cropped)
    return cropped

class CroppedObjectClassificationDataset(Dataset):
    def __init__(self, data_list: List, label2idx: Dict=None, transforms=None, skip_labels=[], is_test=False, random_margin: float = 0.0):
        """
        data_list: List of dicts with 'image' path and 'target' with 'boxes' and 'labels'.
        transforms: Optional transforms applied to each cropped object.
        """
        self.samples = []
        self.transforms = transforms
        self.label2idx = label2idx
        self.random_margin = random_margin

        valid_labels = []
        for entry in data_list:
            labels = entry['target']['labels']
            labels = [x for x in labels if x not in skip_labels]
            valid_labels.extend(labels)

        if label2idx is None:
            self.label2idx = {v: k for k, v in enumerate(set(valid_labels))}
            logger.info(f"Auto infer `label2idx` mapping, mapping length: {len(self.label2idx)}.")

        self.idx2label = {k: v for v, k in self.label2idx.items()}
        self.known_classes = list(self.idx2label.values())
        self.classes = self.known_classes

        for entry in data_list:
            img_path = entry['image']
            boxes = entry['target']['boxes']
            labels = entry['target']['labels']
            task = entry['task']

            for box, label in zip(boxes, labels):
                if label in skip_labels:
                    continue

                labels.append(label)

                if label not in self.label2idx and is_test == True:
                    idx = -1
                else:
                    idx = self.label2idx[label]

                self.samples.append({
                    'task': task,
                    'image': img_path,
                    'box': box,
                    'label': idx
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image']
        box = sample['box']
        label = sample['label']
        cropped = _crop_image(img_path, box, self.transforms, self.random_margin)
        return cropped, label

    def get_labels(self):
        labels = [x["label"] for x in self.samples]
        return labels

class CroppedImageTextDataset(CroppedObjectClassificationDataset):
    def __init__(
        self,
        data_list: List,
        label_desc: dict,
        preprocess_fn: Callable,
        tokenizer,
        label2idx: Dict=None,
        transforms=None,
        skip_labels=[],
        is_test=False,
        random_margin: float=0.0,
        text_template: str="a photo of {desc}"
    ):
        super().__init__(
            data_list=data_list,
            label2idx=label2idx,
            transforms=transforms,
            skip_labels=skip_labels,
            is_test=is_test,
            random_margin=random_margin
        )
        self.text_template = text_template
        self.label_desc = label_desc
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image']
        box = sample['box']
        label: int = sample['label']
        barcode = self.idx2label[label].split("-")[0]
        label_desc = self.label_desc.get(barcode, "object")

        if label_desc == "object":
            logger.warning(f"Unknown found, label={label}, name={self.idx2label[label]}.")

        text = self.text_template.format(desc=label_desc)
        cropped = _crop_image(img_path, box, self.transforms, self.random_margin)
        cropped = self.preprocess_fn(cropped)
        text = self.tokenizer(text)
        return {
            "image": cropped,
            "text": text.squeeze(0)
        }
