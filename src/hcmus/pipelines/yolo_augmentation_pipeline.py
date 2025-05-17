import os
import random
import numpy as np
import cv2
import yaml

from typing import Literal, List, Dict
from loguru import logger
from tqdm import tqdm
from PIL import Image
from hcmus.core import appconfig
from hcmus.lbs import LabelStudioConnector
from hcmus.utils import viz_utils
from hcmus.data import AugmentTemplate


def generate_one_sample(
    augment_template: AugmentTemplate,
    all_backgrounds,
    all_objects,
    n_min_objects: int = 3,
    n_max_objects: int = 10
):
    selected_background = random.choice(all_backgrounds)
    n_objects = random.randint(n_min_objects, n_max_objects)
    selected_objects = []
    selected_labels = []

    for _ in range(n_objects):
        data = random.choice(all_objects)
        selected_objects.append(data.get("object"))
        selected_labels.append(data.get("label"))

    new_background, new_boxes = augment_template.augment(
        image=selected_background.get("background"),
        boxes=selected_background.get("boxes")
    )

    new_sample, fit_boxes, fit_labels = augment_template.place(
        background=new_background,
        boxes=new_boxes,
        objects=selected_objects,
        labels=selected_labels
    )
    return new_sample, fit_boxes, fit_labels


def fetch_objects(project_key: str = "train"):
    result = []
    train_connector = LabelStudioConnector(
        url=appconfig.LABEL_STUDIO_URL,
        api_key=appconfig.LABEL_STUDIO_API_KEY,
        project_id=appconfig.LABEL_STUDIO_PROJECT_MAPPING[project_key],
        temp_dir=appconfig.LABEL_STUDIO_TEMP_DIR
    )
    tasks = train_connector.get_tasks()
    label_dict = train_connector.extract_labels(tasks)
    dataset = train_connector.download_dataset(tasks, label_dict)
    for item in tqdm(dataset, "Extract objects"):
        img = item.get("image")
        img_object = Image.open(img)
        boxes = item.get("target").get("boxes")
        crops = viz_utils.crop_image(img_object, boxes)
        labels = item.get("target").get("labels")
        for i in range(len(boxes)):
            box = boxes[i]
            crop = np.array(crops[i])
            label = labels[i]
            result.append({
                "path": img,
                "box": box,
                "object": crop,
                "label": list(label_dict.keys())[label],
                "label_id": label
            })
    return result, label_dict


def fetch_backgrounds(project_key: str = "template"):
    result = []
    template_connector = LabelStudioConnector(
        url=appconfig.LABEL_STUDIO_URL,
        api_key=appconfig.LABEL_STUDIO_API_KEY,
        project_id=appconfig.LABEL_STUDIO_PROJECT_MAPPING[project_key],
        temp_dir=appconfig.LABEL_STUDIO_TEMP_DIR
    )
    tasks = template_connector.get_tasks()
    dataset = template_connector.download_dataset(tasks)
    for item in dataset:
        img = item.get("image")
        boxes = item.get("target").get("boxes")
        result.append({
            "background": np.array(Image.open(img)),
            "boxes": boxes
        })
    return result


def save_yolo_v8_dataset_from_dicts(
    data: List[Dict],
    class_list: List[str],
    output_dir: str = "dataset",
    split_ratio: float = 0.8
):
    class_to_id = {cls: idx for idx, cls in enumerate(class_list)}

    # Create YOLOv8 folder structure
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

    indices = list(range(len(data)))
    random.shuffle(indices)
    split = int(len(data) * split_ratio)

    for count, idx in enumerate(indices):
        item = data[idx]
        image = item['image']
        boxes = item['target']['boxes']
        labels = item['target']['labels']

        h, w = image.shape[:2]
        split_type = 'train' if count < split else 'val'

        filename = f"{idx:05d}.jpg"
        img_path = os.path.join(output_dir, f"images/{split_type}", filename)
        label_path = os.path.join(output_dir, f"labels/{split_type}", filename.replace('.jpg', '.txt'))

        # Save image
        cv2.imwrite(img_path, image)

        # Save label
        with open(label_path, 'w') as f:
            for box, label in zip(boxes, labels):
                if label not in class_to_id:
                    continue
                class_id = class_to_id[label]
                x_min, y_min, x_max, y_max = box

                # Normalize coordinates
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                bbox_width = (x_max - x_min) / w
                bbox_height = (y_max - y_min) / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    # Save data.yaml
    yaml_dict = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_list)}
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    logger.info(f"YOLOv8-compatible dataset saved to: {output_dir}")


def execute(
    output_dir: str,
    output_format: Literal["yolo"] = "yolo",
    split_ratio: float = 0.8,
    background_project_key: str = "template",
    object_project_key: str = "train",
    n_augment: int = 1000,
):
    augment_template = AugmentTemplate()
    all_backgrounds = fetch_backgrounds(background_project_key)
    all_objects, label_dict = fetch_objects(object_project_key)
    dataset = []

    for _ in tqdm(range(n_augment), "Augmenting"):
        image, boxes, labels = generate_one_sample(augment_template, all_backgrounds, all_objects)
        dataset.append({
            "image": image,
            "target": {
                "boxes": boxes,
                "labels": labels
            }
        })

    if output_format == "yolo":
        save_yolo_v8_dataset_from_dicts(dataset, list(label_dict.keys()), output_dir, split_ratio=split_ratio)
    else:
        raise ValueError("Only accept `yolo`.")
