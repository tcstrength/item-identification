import numpy as np
import timm
import faiss
import torch
import clip
import mlflow
from tqdm import tqdm
from loguru import logger
from torch.nn import functional as F


class MatchingNetwork(nn.Module):
    def __init__(self, backbone_name: str, support_images, support_labels, mlflow_logged_model: str = None):
        super().__init__()
        if mlflow_logged_model is not None:
            self.model = mlflow.pyfunc.load_model(mlflow_logged_model).get_raw_model()
        else:
            self.model, _ = clip.load(backbone_name, device="cuda" if torch.cuda.is_available() else "cpu")

        self.visual_encoder = self.model.visual

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        if "ViT-B" in backbone_name:
            self.feature_dim = 512
        elif "ViT-L" in backbone_name:
            self.feature_dim = 768
        elif "RN50" in backbone_name:
            self.feature_dim = 1024
        elif "RN101" in backbone_name:
            self.feature_dim = 512

        self.support_labels = support_labels
        self.index = faiss.IndexHNSWFlat(self.emb_size, 128)
        for item in tqdm(support_images, desc="Building index"):
            # item = item.unsqueeze(0)
            emb = self.create_embeddings(item)
            # print(emb.shape, self.emb_size)
            self.index.add(emb)

        logger.info(f"Index shape: {self.index.ntotal}")

    def freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def create_embeddings(self, images):
        if self.is_hf:
            self.backbone.eval()
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt")
                embs = self.backbone.get_image_features(**inputs)
        else:
            self.backbone.eval()
            embs = self.backbone(images)

        embs = F.normalize(embs, dim=-1)
        embs = embs.detach().numpy().astype("float32")
        return embs

    def predict(self, query_images, k: int = 3):
        result = []
        for q in query_images:
            q = self.create_embeddings(q)
            D, I = self.index.search(q.reshape(1, -1), k=k)
            result.append({
                "D": D[0],
                "I": I[0],
                "L": [self.support_labels[i].item() for i in I[0]],
            })

        return result
