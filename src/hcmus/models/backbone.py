import clip
import mlflow
from abc import abstractmethod
from torch import nn


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
    def __init__(self, backbone_name: str, device: str="cpu"):
        super().__init__()
        if self.is_mlflow(backbone_name):
            model = mlflow.pyfunc.load_model(backbone_name).get_raw_model()
        else:
            model, _ = clip.load(backbone_name, device=device)

        self._visual_encoder = model.visual
        self.freeze(self._visual_encoder)

    def forward(self, x):
        # Consider as batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        self._visual_encoder.eval()
        return self._visual_encoder(x)

    @property
    def output_dim(self):
        return self._visual_encoder.output_dim


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
