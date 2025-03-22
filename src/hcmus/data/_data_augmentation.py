import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict
from PIL import Image

class DataAugmentation:
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        ])

    def __call__(self, image: Image, target: Dict):
        boxes = torch.tensor(target["boxes"], dtype=torch.float32)
        orig_w, orig_h = image.size

        for t in self.transforms.transforms:
            if isinstance(t, T.RandomHorizontalFlip) and random.random() < t.p:
                image = F.hflip(image)
                boxes[:, [0, 2]] = orig_w - boxes[:, [2, 0]]

            elif isinstance(t, T.RandomResizedCrop):
                i, j, h, w = T.RandomResizedCrop.get_params(image, scale=t.scale, ratio=(1.0, 1.0))
                image = F.resized_crop(image, i, j, h, w, self.img_size)

                # Adjust bounding boxes
                boxes[:, [0, 2]] = boxes[:, [0, 2]] - j
                boxes[:, [1, 3]] = boxes[:, [1, 3]] - i
                boxes[:, [0, 2]] *= self.img_size[1] / w
                boxes[:, [1, 3]] *= self.img_size[0] / h

            elif isinstance(t, T.RandomRotation):
                angle = random.uniform(t.degrees[0], t.degrees[1])
                image = F.rotate(image, angle)
                boxes = self.rotate_boxes(boxes, angle, orig_w, orig_h)
            else:
                image = t(image)

        target["boxes"] = boxes.tolist()
        return image, target

    def rotate_boxes(self, boxes, angle, img_w, img_h):
        """
        Rotates bounding boxes around the image center.
        """
        angle = torch.tensor(angle).float().deg2rad()
        cx, cy = img_w / 2, img_h / 2  # Image center
        new_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            corners = torch.tensor([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_min],
                [x_max, y_max]
            ])

            # Translate to origin (center of image)
            corners[:, 0] -= cx
            corners[:, 1] -= cy

            # Apply rotation matrix
            rotation_matrix = torch.tensor([
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)]
            ])
            rotated_corners = torch.mm(corners, rotation_matrix.T)

            # Translate back
            rotated_corners[:, 0] += cx
            rotated_corners[:, 1] += cy

            # Get new bounding box
            x_min, y_min = rotated_corners.min(dim=0)[0]
            x_max, y_max = rotated_corners.max(dim=0)[0]
            new_boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

        return torch.tensor(new_boxes)
