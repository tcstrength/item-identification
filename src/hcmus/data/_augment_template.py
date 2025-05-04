import cv2
import numpy as np
import random
from typing import List, Tuple

class AugmentTemplate:
    def __init__(
        self,
        rotation: Tuple[int] = (-15, 15),
        zoom: Tuple[float] = (0.8, 2.5),
        hflip: bool = True
    ):
        self._rotation = rotation
        self._hflip = hflip
        self._zoom = zoom

    def __rotate_image_and_boxes(self, image: np.ndarray, boxes: List[Tuple[int]], angle: int):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

        def rotate_box(box):
            x_min, y_min, x_max, y_max = box
            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
            ones = np.ones((4, 1))
            corners_ = np.hstack([corners, ones])
            rotated_corners = M.dot(corners_.T).T

            x_coords = rotated_corners[:, 0]
            y_coords = rotated_corners[:, 1]
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        rotated_boxes = [rotate_box(box) for box in boxes]
        return rotated_image, rotated_boxes

    def __flip_image_and_boxes_horizontally(self, image: np.ndarray, boxes: List[Tuple[int]]):
        flipped_image = np.fliplr(image).copy()
        h, w = image.shape[:2]

        flipped_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            new_x_min = w - x_max
            new_x_max = w - x_min
            flipped_boxes.append((new_x_min, y_min, new_x_max, y_max))

        return flipped_image, flipped_boxes

    def __random_zoom(self, image: np.ndarray, boxes: List[Tuple[int]], zoom_factor: float):
        h, w = image.shape[:2]

        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        # Resize image
        zoomed_image = cv2.resize(image, (new_w, new_h))
        zoomed_boxes = []

        if zoom_factor < 1.0:
            # Pad and center
            pad_w = (w - new_w) // 2
            pad_h = (h - new_h) // 2
            padded = np.zeros_like(image)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = zoomed_image
            zoomed_image = padded

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_min = x_min * zoom_factor + pad_w
                x_max = x_max * zoom_factor + pad_w
                y_min = y_min * zoom_factor + pad_h
                y_max = y_max * zoom_factor + pad_h

                if x_min >= 0 and y_min >= 0 and x_max <= w and y_max <= h:
                    zoomed_boxes.append((x_min, y_min, x_max, y_max))

        else:
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            zoomed_image = zoomed_image[crop_y:crop_y + h, crop_x:crop_x + w]

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_min = x_min * zoom_factor - crop_x
                x_max = x_max * zoom_factor - crop_x
                y_min = y_min * zoom_factor - crop_y
                y_max = y_max * zoom_factor - crop_y

                if x_min >= 0 and y_min >= 0 and x_max <= w and y_max <= h:
                    zoomed_boxes.append((x_min, y_min, x_max, y_max))

        return zoomed_image, zoomed_boxes


    def augment(self, image: np.array, boxes: List[Tuple[int]]):
        do_zoom = random.randint(0, 1)
        do_rotation = random.randint(0, 1)
        do_hflip = random.randint(0, 1)

        if do_zoom > 0:
            zoom_factor = random.uniform(self._zoom[0], self._zoom[1])
            image, boxes = self.__random_zoom(image, boxes, zoom_factor)

        if do_rotation > 0:
            angle = random.randint(self._rotation[0], self._rotation[1])
            image, boxes = self.__rotate_image_and_boxes(image, boxes, angle)

        if self._hflip and do_hflip > 0:
            image, boxes = self.__flip_image_and_boxes_horizontally(image, boxes)

        return image, boxes


    def place(self, background: np.array, boxes: List[Tuple[int]], objects: List[np.array]):
        bg = background.copy()
        bg_h, bg_w = bg.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            bw, bh = x2 - x1, y2 - y1

            obj = random.choice(objects)

            oh, ow = obj.shape[:2]

            # Preserve aspect ratio: scale by the smaller fitting factor
            scale = min(bw / ow, bh / oh)
            new_w, new_h = max(1, int(ow * scale)), max(1, int(oh * scale))
            resized_obj = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Placement
            start_x = int(x1 + (bw - new_w) / 2)
            start_y = int(y1 + (bh - new_h) / 2)
            end_x, end_y = start_x + new_w, start_y + new_h

            if start_x < 0 or start_y < 0 or end_x > bg_w or end_y > bg_h:
                continue

            bg_patch = bg[start_y:end_y, start_x:end_x]

            if resized_obj.shape[2] == 4:
                alpha = resized_obj[:, :, 3].astype(np.float32) / 255.0
                for c in range(3):
                    bg[start_y:end_y, start_x:end_x, c] = (
                        alpha * resized_obj[:, :, c] +
                        (1 - alpha) * bg[start_y:end_y, start_x:end_x, c]
                    )
            else:
                bg[start_y:end_y, start_x:end_x] = resized_obj

        return bg
