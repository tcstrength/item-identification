import numpy as np
import timm
import faiss
import torch
import clip
import mlflow
from torch import nn
from tqdm import tqdm
from loguru import logger
from torch.nn import functional as F


class SearchModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "ViT-B/32",
        index_dim: int = 256,
        mlflow_logged_model: str = None,
        faiss_path: str = None
    ):
        super().__init__()
        if mlflow_logged_model is not None:
            self.model = mlflow.pyfunc.load_model(mlflow_logged_model).get_raw_model()
        else:
            self.model, _ = clip.load(backbone_name, device="cuda" if torch.cuda.is_available() else "cpu")

        self.visual_encoder = self.model.visual
        self.support_images = []
        self.support_labels = []

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

        if faiss_path is not None:
            self.index = faiss.read_index(faiss_path)
        else:
            self.index = faiss.IndexHNSWFlat(self.feature_dim, index_dim)

    def add_support_set(self, support_images, support_labels):
        self.support_labels.extend(support_labels)
        self.support_images.extend(support_images)
        embs = self.create_embeddings(support_images)
        self.index.add(embs)


    def create_embeddings(self, images):
        self.visual_encoder.eval()
        embs = self.visual_encoder(images)
        embs = F.normalize(embs, dim=-1)
        embs = embs.detach().numpy().astype("float32")
        return embs

    def save_index(self, faiss_path):
        faiss.write_index(self.index, faiss_path)

    def forward(self, query_images, k: int = 3):
        if len(query_images.shape) == 3:
            query_images = query_images.unsqueeze(0)

        result = []
        embs = self.create_embeddings(query_images)
        D, I = self.index.search(embs, k=k)
        for k in range(len(D)):
            result.append({
                "D": D[k],
                "I": I[k],
                "L": [self.support_labels[i].item() for i in I[k]],
            })

        return result
