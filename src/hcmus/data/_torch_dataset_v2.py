import random
import torch
import copy
from typing import Callable, Dict, List
from loguru import logger
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def _crop_image(img_path, box, transforms, random_margin: float = 0.2):
    box = map(int, box)
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    max_w, max_h = image.size

    x1, y1, x2, y2 = map(int, box)
    margin_w = 0
    margin_h = 0

    if random_margin != 0:
        margin_w = (x2 - x1) * (random.random() - 0.5) * random_margin / 2
        margin_h = (y2 - y1) * (random.random() - 0.5) * random_margin / 2
        x1 = max(x1 - margin_w, 0)
        y1 = max(y1 - margin_h, 0)
        x2 = min(x2 + margin_w, max_w)
        y2 = min(y2 + margin_w, max_h)

    try:
        cropped = image.crop((x1, y1, x2, y2))
    except Exception as e:
        logger.warning(f"{e}, given ({x1},{y1},{x2},{y2})")
        return None

    if transforms:
        cropped = transforms(cropped)
    return cropped


class CroppedImageDataset(Dataset):
    def __init__(
        self,
        data_list: List,
        label2idx: Dict=None,
        transforms=None,
        skip_labels=[],
        ignore_unknown=True,
        random_margin: float = 0.0
    ):
        self.samples = []
        self.transforms = transforms
        self.label2idx = label2idx
        self.random_margin = random_margin

        logger.info(f"Apply random_margin={self.random_margin}")

        valid_labels = []
        data_list = copy.deepcopy(data_list)
        for entry in data_list:
            labels = entry["target"]["labels"]
            labels = [x for x in labels if x not in skip_labels]
            valid_labels.extend(labels)

        if label2idx is None:
            self.label2idx = {v: k for k, v in enumerate(set(valid_labels))}
            logger.info(f"Auto infer `label2idx` mapping, mapping length: {len(self.label2idx)}.")

        self.idx2label = {k: v for v, k in self.label2idx.items()}
        self.classes = list(self.idx2label.values())

        for entry in data_list:
            img_path = entry["image"]
            boxes = entry["target"]["boxes"]
            labels = entry["target"]["labels"]
            task = entry["task"]

            for box, label in zip(boxes, labels):
                if label in skip_labels:
                    continue

                # labels.append(label)

                if label not in self.label2idx and ignore_unknown == False:
                    idx = -1
                else:
                    idx = self.label2idx[label]

                # if label == "object":
                #     print(idx)

                self.samples.append({
                    "task": task,
                    "image": img_path,
                    "box": box,
                    "label_idx": idx,
                    "label_str": self.idx2label[idx]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image"]
        box = sample["box"]
        label_idx = sample["label_idx"]
        label_str = sample["label_str"]
        task = sample["task"]
        cropped = _crop_image(img_path, box, self.transforms, self.random_margin)
        metadata = {
            "label_str": label_str,
            "task_id": task.id,
            "box": box,
            "path": img_path
        }
        return cropped, label_idx, metadata

    def collate_fn(self, batch):
        data = [item[0] for item in batch]  # Extract data
        labels = [item[1] for item in batch]  # Extract labels

        data = torch.stack(data)
        labels = torch.tensor(labels)

        return data, labels

    def get_labels(self):
        labels = [x["label"] for x in self.samples]
        return labels


class CroppedImageTextDataset(CroppedImageDataset):
    def __init__(
        self,
        data_list,
        product_desc: dict,
        preprocess_fn: Callable,
        tokenizer,
        text_template: str = "{desc}",
        **kwargs
    ):
        super().__init__(data_list, **kwargs)
        self.label_desc =product_desc
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.text_template = text_template

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image']
        box = sample['box']
        label: int = sample['label_idx']
        barcode = self.idx2label[label].split("-")[0]
        desc_ls = self.label_desc.get(barcode, ["A random object"])
        label_desc = random.choice(desc_ls)

        if label_desc == "A random object":
            logger.warning(f"Unknown found, label={label}, name={self.idx2label[label]}.")

        text = self.text_template.format(desc=label_desc)
        cropped = _crop_image(img_path, box, self.transforms, self.random_margin)
        cropped = self.preprocess_fn(cropped)
        text = self.tokenizer(text)
        return {
            "image": cropped,
            "text": text.squeeze(0)
        }
