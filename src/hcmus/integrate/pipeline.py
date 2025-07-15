from loguru import logger
from typing import Union
from PIL import Image
from PIL import ImageOps
from torch import nn
from torchvision import transforms as T
from hcmus.integrate.rpn import RegionProposal
from hcmus.integrate.classifier import Classifier

class ModelPipeline:
    def __init__(self, rpn: RegionProposal, classifier: Classifier, transforms: T.Compose):
        self.rpn = rpn
        self.classifier = classifier
        self.transforms = transforms

    def forward(
        self,
        image: Union[Image.Image, str],
        region_threshold: float = 0.5,
        unknown_method: str = "entropy",
        unknown_threshold: float = 0.49
    ):
        if isinstance(image, str):
            image = Image.open(image)
            image = ImageOps.exif_transpose(image)

        boxes = self.rpn.forward(image, region_threshold)
        boxes = [list(map(int, x)) for x in boxes]
        logger.info(f"Removing duplicated boxes, before: {len(boxes)}")
        boxes = self.rpn.remove_duplicates(boxes)
        logger.info(f"Deduplicated boxes: {len(boxes)}")

        subs = [image.crop(x) for x in boxes]
        subs = [self.transforms(x) for x in subs]
        result = []

        for box, sub in zip(boxes, subs):
            logits = self.classifier.forward(sub)
            is_unknown, score = self.classifier.detect_unknown(
                logits=logits,
                method=unknown_method,
                threshold=unknown_threshold
            )
            pred_idx = logits.argmax().item()
            if is_unknown:
                pred_idx = -1

            pred_label = self.classifier.idx2label.get(pred_idx)

            result.append({
                "box": box,
                "pred_idx": pred_idx,
                "pred_label": pred_label,
                "unknown_method": unknown_method,
                "unknown_threshold": unknown_threshold,
                "unknown_score": score
            })
        return result
