import json
import torch
import mlflow
import mlflow.artifacts
from torch.nn import functional as F
from typing import Literal

class Classifier():
    def __init__(self, run_id: str):
        run = mlflow.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        label2idx = mlflow.artifacts.load_text(f"{artifact_uri}/json/label2idx.json")
        self.label2idx = json.loads(label2idx)
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model").get_raw_model()

    @staticmethod
    def compute_entropy(probs):
        log_probs = torch.log(probs + 1e-12)  # Avoid log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    @staticmethod
    def detect_unknown(logits, method: Literal["entropy", "prob"], threshold: float):
        probs = F.softmax(logits, dim=-1)
        if method == "prob":
            score = probs.max().item()
            return score < threshold, score
        elif method == "entropy":
            score = Classifier.compute_entropy(probs).item()
            return score > threshold, score
        else:
            raise ValueError(f"Invalid method {method}")

    def forward(self, image):
        logits = self.model(image)
        return logits


if __name__ == "__main__":
    run_id = "415b1f0c9b8640d9848e0fc470528ecc"
    classifier = Classifier(run_id=run_id)
    from torch.nn import functional as F
    from torchvision import transforms as T
    from PIL import Image

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    image = Image.open("/Volumes/Cucumber/Projects/datasets/raw/hcmus-iid/train/images/df898a0bd64bb55977029e5c2c4c8533.jpg")
    image = transforms(image)
    logits = classifier.forward(image)
    probs = F.softmax(logits, dim=-1)
    entropy = classifier.compute_entropy(probs)
    probs.max()

