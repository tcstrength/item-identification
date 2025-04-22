import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
from pydantic import BaseModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from transformers import CLIPProcessor, CLIPModel
from PIL.Image import Image
from hcmus.models._base_model import ModelOutput


UNKNOWN_LABEL = "unknown"
NUM_PREDICTION = 1


class kNNSearchConfig(BaseModel):
    images: List[Image]
    labels: List[str]
    augmentation: bool = False
    embedding_dim: int = 512
    embedding_model: str = "CLIP"
    box_threshold: float = 0.1
    label_threshold: float = 60

    class Config:
        arbitrary_types_allowed = True


class kNNSearchModel:
    def __init__(self, config: kNNSearchConfig):
        self._config = config

        l1 = len(self._config.images)
        l2 = len(self._config.labels)
        if l1 != l2:
            raise ValueError(f"Number of images must be equal to number of labels: {l1} != {l2}")

        self._rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self._rcnn_model.eval()

        if self._config.embedding_model == "CLIP":
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        else:
            raise ValueError("Only support embedding_model=CLIP")

        self._faiss_index = self._build_index(self._config.images, self._config.embedding_dim)
        self._to_tensor = ToTensor()

    def _get_embedding(self, image: Image) -> np.array:
        inputs = self._clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._clip_model.get_image_features(**inputs)
        return outputs[0].cpu().numpy()

    def _build_index(self, images: List[Image], dim: int, num_workers: int = None) -> faiss.Index:
        index = faiss.IndexFlatL2(dim)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create a list to store futures
            futures = [executor.submit(self._get_embedding, image) for image in images]

            # Process results with tqdm for progress tracking
            embeddings = []
            for future in tqdm(futures, total=len(futures), desc="Building index..."):
                result = future.result()
                if result is None: continue
                embeddings.append(result)

        # Add all embeddings to the index at once
        index.add(np.array(embeddings))
        return index

    def _crop_image(self, image: Image, boxes: List):
        sub_images = []
        for box in boxes:
            box = [int(coord) for coord in box]
            sub_img = image.crop(box)
            sub_images.append(sub_img)
        return sub_images

    def _distance_to_score(self, distance: float, scale: float = 1.0):
        return np.exp(-distance / scale)

    def _assign_label(self, image: Image):
        query_embedding = self._get_embedding(image)
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self._faiss_index.search(query_embedding, NUM_PREDICTION)
        label = UNKNOWN_LABEL
        score = distances[0][0]

        if score < self._config.label_threshold:
            label = self._config.labels[indices[0][0]]

        return label, score

    def _get_boxes(self, image: Image):
        tensor = self._to_tensor(image)
        output = self._rcnn_model([tensor])[0]
        boxes = []
        scores = []
        for box, score in zip(output["boxes"], output["scores"]):
            if score >= self._config.box_threshold:
                box = [int(coord) for coord in box]
                boxes.append(box)
                scores.append(score)
        return boxes, scores

    def predict(self, image: Image) -> List[ModelOutput]:
        result = []
        boxes, scores = self._get_boxes(image)
        subs = self._crop_image(image, boxes)
        for box, box_score, sub in zip(boxes, scores, subs):
            label, distance_score = self._assign_label(sub)
            result.append(ModelOutput(
                box=box,
                label=label,
                metadata={
                    "box_score": box_score,
                    "distance_score": distance_score
                }
            ))
        return result

