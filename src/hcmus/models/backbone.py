import clip
import mlflow
import torch
from abc import abstractmethod
from torch import nn
from transformers import AutoModel, AutoImageProcessor


class BaseBackbone(nn.Module):
    @property
    def output_dim(self) -> int:
        pass

    @staticmethod
    def freeze(nn):
        for param in nn.parameters():
            param.requires_grad = False

    @staticmethod
    def is_mlflow(backbone_name: str):
        return backbone_name.startswith("runs:/")


class CLIPBackbone(BaseBackbone):
    def __init__(self, backbone_name: str = "ViT-B/32", device: str="cpu"):
        super().__init__()
        if self.is_mlflow(backbone_name):
            model = mlflow.pyfunc.load_model(backbone_name).get_raw_model()
        else:
            model, _ = clip.load(backbone_name, device=device)

        self._visual_encoder = model.visual
        self.freeze(self._visual_encoder)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self._visual_encoder.eval()
            features = self._visual_encoder(x)
        return features

    @property
    def output_dim(self):
        return self._visual_encoder.output_dim


class DinoBackbone(BaseBackbone):
    def __init__(self, model_id: str = "facebook/dinov2-small"):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)
        except:
            self.model = AutoModel.from_pretrained(model_id)
        # self.processor = AutoImageProcessor.from_pretrained(model_name)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(x)
            features = outputs.last_hidden_state[:, 0]
        return features

    @property
    def output_dim(self):
        return self.model.config.hidden_size


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms as T
    backbone_name ="ViT-B/32"
    backbone = CLIPBackbone(backbone_name)
    to_tensor = T.ToTensor()
    resize = T.Resize((224, 224))
    path = "/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0a0c8522300b70f18d35c9393bd69711.jpg"
    image = Image.open(path)
    image = to_tensor(image)
    image = resize(image)
    image = image.unsqueeze(0)
    feature = backbone(image)

    backbone = DinoBackbone()

    to_tensor = T.ToTensor()
    resize = T.Resize((224, 224))
    path = "/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0a0c8522300b70f18d35c9393bd69711.jpg"
    image = Image.open(path)
    image = to_tensor(image)
    image = resize(image)
    image = image.unsqueeze(0)

    feature = backbone(image)
    feature
