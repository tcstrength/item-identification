import os
import json
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
        random_margin: float=0.0,
        return_metadata: bool=True
    ):
        self.samples = []
        self.transforms = transforms
        self.label2idx = label2idx
        self.random_margin = random_margin
        self.return_metadata = return_metadata

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
        if self.return_metadata:
            return cropped, label_idx, metadata
        else:
            return cropped, label_idx

    def collate_fn(self, batch):
        data = [item[0] for item in batch]  # Extract data
        labels = [item[1] for item in batch]  # Extract labels

        data = torch.stack(data)
        labels = torch.tensor(labels)

        return data, labels

    def get_labels(self):
        labels = [x["label_idx"] for x in self.samples]
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


class CroppedCocoDataset(Dataset):
    def __init__(self, split_path, transform=None):
        # Get annotation json
        ann_path = None
        for file in os.listdir(split_path):
            if file.startswith("annotations_") and file.endswith(".json"):
                ann_path = os.path.join(split_path, file)
                break
        assert ann_path is not None, "Annotation file not found!"

        with open(ann_path, 'r') as f:
            coco = json.load(f)

        # Build category id to index mapping
        self.cat2idx = {}
        for idx, cat in enumerate(coco['categories']):
            if cat['name'] == "unknown":
                logger.info(f"Skip category_id={cat['id']}")
                continue

            self.cat2idx[cat['id']] = idx

        # Build image id to filename mapping
        self.imgs = {img['id']: img for img in coco['images']}
        self.img_dir = os.path.join(split_path, "images")

        # For each annotation, store (img_file, bbox, label_idx)
        self.samples = []
        for ann in coco['annotations']:
            img_info = self.imgs[ann['image_id']]
            img_file = os.path.join(self.img_dir, img_info['file_name'])
            bbox = ann['bbox']  # [x, y, w, h]
            label_idx = self.cat2idx.get(ann['category_id'])

            if label_idx is not None:
                self.samples.append((img_file, bbox, label_idx))
            else:
                logger.debug(f"Skip object with category_id={ann['category_id']}")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, bbox, label_idx = self.samples[idx]
        image = Image.open(img_file).convert("RGB")

        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cropped = image.crop((x1, y1, x2, y2))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label_idx

    @property
    def num_classes(self):
        return len(self.cat2idx)
