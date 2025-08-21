import clip
import mlflow
import torch
import torchvision.models as models
from loguru import logger
from torch import nn
from transformers import AutoModel


class BaseBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self._backbone_name = backbone_name

    @property
    def output_dim(self) -> int:
        pass

    def freeze(self, nn):
        logger.info(f"Freeze backbone {self._backbone_name}.")
        for param in nn.parameters():
            param.requires_grad = False

    @staticmethod
    def is_mlflow(backbone_name: str):
        return backbone_name.startswith("runs:/")


class CLIPBackbone(BaseBackbone):
    def __init__(self, backbone_name: str = "ViT-B/32", device: str="cpu", freeze: bool=True):
        super().__init__(backbone_name)

        if backbone_name not in ("ViT-B/32", "ViT-B/16"):
            raise ValueError(f"Unsupported CLIP variant (ViT-B/32, ViT-B/16): {backbone_name}")

        if self.is_mlflow(backbone_name):
            model = mlflow.pyfunc.load_model(backbone_name).get_raw_model()
        else:
            model, _ = clip.load(backbone_name, device=device)

        self._visual_encoder = model.visual
        if freeze:
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
    def __init__(self, backbone_name: str = "facebook/dinov2-small", freeze: bool=True):
        super().__init__(backbone_name)

        if backbone_name not in ("facebook/dinov2-small", "facebook/dinov2-base"):
            raise ValueError(f"Unsupported CLIP variant (facebook/dinov2-small, facebook/dinov2-base): {backbone_name}")

        try:
            self.model = AutoModel.from_pretrained(backbone_name, add_pooling_layer=False)
        except:
            self.model = AutoModel.from_pretrained(backbone_name)

        if freeze:
            self.freeze(self.model)

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


class ResNetBackbone(BaseBackbone):
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, device: str = "cpu", freeze: bool=True):
        super().__init__(backbone_name)

        if self.is_mlflow(backbone_name):
            model = mlflow.pyfunc.load_model(backbone_name).get_raw_model()
            self._feature_extractor = model
        else:
            # Load pretrained ResNet model
            if backbone_name == "resnet50":
                model = models.resnet50(pretrained=pretrained)
                self._output_dim = 2048
            elif backbone_name == "resnet101":
                model = models.resnet101(pretrained=pretrained)
                self._output_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant (resnet50, resnet101): {backbone_name}")

            # Remove the final classification layer (fc) to use as feature extractor
            self._feature_extractor = nn.Sequential(*list(model.children())[:-1])

        self._feature_extractor.to(device)
        if freeze:
            self.freeze(self._feature_extractor)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self._feature_extractor.eval()
            features = self._feature_extractor(x)
            # Flatten the spatial dimensions (batch_size, channels, 1, 1) -> (batch_size, channels)
            features = features.view(features.size(0), -1)

        return features

    @property
    def output_dim(self):
        return self._output_dim


class DenseNetBackbone(BaseBackbone):
    def __init__(self, backbone_name: str = "densenet121", pretrained: bool = True, device: str = "cpu", freeze: bool=True):
        super().__init__(backbone_name)

        if self.is_mlflow(backbone_name):
            model = mlflow.pyfunc.load_model(backbone_name).get_raw_model()
            self._feature_extractor = model
        else:
            # Load pretrained DenseNet model
            if backbone_name == "densenet121":
                model = models.densenet121(pretrained=pretrained)
                self._output_dim = 1024
            elif backbone_name == "densenet169":
                model = models.densenet169(pretrained=pretrained)
                self._output_dim = 1664
            else:
                raise ValueError(f"Unsupported DenseNet variant (densenet121, densenet169): {backbone_name}")

            # Remove the final classification layer to use as feature extractor
            self._feature_extractor = model.features
            # Add adaptive pooling to get consistent output size
            self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_extractor.to(device)
        self._adaptive_pool.to(device)
        if freeze:
            self.freeze(self._feature_extractor)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self._feature_extractor.eval()
            features = self._feature_extractor(x)
            # Apply ReLU activation (DenseNet features don't include final ReLU)
            features = torch.relu(features)
            # Apply adaptive pooling
            features = self._adaptive_pool(features)
            # Flatten the spatial dimensions
            features = features.view(features.size(0), -1)

        return features

    @property
    def output_dim(self):
        return self._output_dim


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
    feature.shape #(1,512)

    backbone = DinoBackbone()
    to_tensor = T.ToTensor()
    resize = T.Resize((224, 224))
    path = "/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0a0c8522300b70f18d35c9393bd69711.jpg"
    image = Image.open(path)
    image = to_tensor(image)
    image = resize(image)
    image = image.unsqueeze(0)
    feature = backbone(image)
    feature.shape #(1,384)

    backbone = ResNetBackbone()
    to_tensor = T.ToTensor()
    resize = T.Resize((224, 224))
    path = "/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0a0c8522300b70f18d35c9393bd69711.jpg"
    image = Image.open(path)
    image = to_tensor(image)
    image = resize(image)
    image = image.unsqueeze(0)
    feature = backbone(image)
    feature.shape #(1,2048)


    backbone = DenseNetBackbone()
    to_tensor = T.ToTensor()
    resize = T.Resize((224, 224))
    path = "/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid-lbs/0a0c8522300b70f18d35c9393bd69711.jpg"
    image = Image.open(path)
    image = to_tensor(image)
    image = resize(image)
    image = image.unsqueeze(0)
    feature = backbone(image)
    feature.shape #(1,1024)
