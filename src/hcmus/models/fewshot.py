import numpy as np
import timm
import faiss
import torch
from tqdm import tqdm
from loguru import logger
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel


class MatchingNetwork():
    def __init__(self, backbone_name: str, support_images, support_labels, is_hf: bool=False, emb_size: int=None):
        if is_hf:
            self.backbone = CLIPModel.from_pretrained(backbone_name)
            self.processor = CLIPProcessor.from_pretrained(backbone_name)
            # self.emb_size = self.backbone.vision_model.config.hidden_size
            if emb_size is None:
                self.emb_size = self.backbone.vision_model.config.hidden_size
            else:
                self.emb_size = emb_size
        else:
            self.backbone = timm.create_model(
                model_name=backbone_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            self.emb_size = self.backbone.num_features
            self.freeze_backbone(self.backbone)

        self.is_hf = is_hf
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
        # queries = []
        # for q in query_images:
        #     queries.append(self.create_embeddings(q.unsqueeze(0)))
        # queries = self.create_embeddings(query_images)
        for q in query_images:
            q = self.create_embeddings(q)
            D, I = self.index.search(q.reshape(1, -1), k=k)
            result.append({
                "D": D[0],
                "I": I[0],
                "L": [self.support_labels[i].item() for i in I[0]],
            })

        return result
