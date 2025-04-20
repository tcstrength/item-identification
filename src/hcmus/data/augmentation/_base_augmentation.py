import torch
from typing import Dict
from PIL.Image import Image

class BaseAugmentation:
    def __call__(self, image: Image, target: Dict[str, torch.Tensor]):
        raise NotImplementedError()
