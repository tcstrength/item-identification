"""
%load_ext autoreload
%autoreload 2
"""
from collections import Counter
from turtle import title
from tqdm import tqdm
from hcmus.utils import data_utils, viz_utils
from hcmus.utils import transform_utils
from hcmus.models.search import SearchModel

splits = data_utils.get_data_splits()
transform_train, transform_test = transform_utils.get_transforms_downscale(128)
datasets = data_utils.get_image_datasets_v2(splits, transform_train, transform_test)


dataloaders = data_utils.get_data_loaders_v2(datasets, {"train": True})

# faiss_path = "index.faiss"
top_k = 5
model = SearchModel(
    mlflow_logged_model="runs:/16c530d050a24b7095a3bd2585f2c789/model"
)

ensemble_epochs = 3
model = SearchModel(mlflow_logged_model="runs:/16c530d050a24b7095a3bd2585f2c789/model")
for _ in range(ensemble_epochs):
    for batch in tqdm(dataloaders["train"], "Ensembling..."):
        images, labels = batch
        model.create_embeddings(images).shape
        model.add_support_set(images, labels)
# model.save_index(faiss_path)

result_records = []
for item in tqdm(datasets["test"], "Evaluating..."):
    img, label, metadata = item
    result = model.forward(img, k=top_k)[0]
    counter = Counter(result.get("L"))
    pred = counter.most_common(1)[0][0]
    result_records.append({
        "pred": pred,
        "label": label,
        "task_id": metadata.get("task_id"),
        "box": metadata.get("box"),
        "path": metadata.get("path"),
        "label_str": metadata.get("label_str"),
        "pred_str": datasets["test"].idx2label[pred]
    })

result_records

import pandas as pd
df = pd.DataFrame(result_records)
df["correct"] = df["pred"] == df["label"]
df_agg = df.groupby("task_id").agg(
    correct=pd.NamedAgg(column="correct", aggfunc="sum"),
    count=pd.NamedAgg(column="correct", aggfunc="count"),
)

df_agg["accuracy"] = df_agg["correct"] / df_agg["count"]
df_agg[df_agg["accuracy"] < 0.5]["correct"].sum() / df_agg[df_agg["accuracy"] > 0.5]["count"].sum()
df_agg[df_agg["accuracy"] >= 0.5]["correct"].sum() / df_agg[df_agg["accuracy"] > 0.5]["count"].sum()

df_hard = df_agg[df_agg["accuracy"] < 0.5].reset_index()
df_hard["segment"] = "hard"
df_hard = df_hard[["task_id", "segment", "accuracy"]]
df_easy = df_agg[df_agg["accuracy"] >= 0.5].reset_index()
df_easy["segment"] = "easy"
df_easy = df_easy[["task_id", "segment", "accuracy"]]
df_segment = pd.concat([df_hard, df_easy])
df_segment
df = pd.merge(df, df_segment, on="task_id")

from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageOps

excluded_id = [
    2020, 2513, 4963, 2375, 4962, 2028
]
df[(~df["task_id"].isin(excluded_id)) & (df["correct"] == False)][["task_id", "correct", "label_str", "pred_str"]]

def show_img_from_df(df, idx: int):
    task_id = df[df.index == idx]["task_id"].item()
    x1, y1, x2, y2 = df[df.index == idx]["box"].item()
    path = df[df.index == idx]["path"].item()
    true = df[df.index == idx]["label_str"].item()
    pred = df[df.index == idx]["pred_str"].item()
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = viz_utils.crop_image(img, [(x1, y1, x2, y2)])[0]
    plt.imshow(img)
    plt.title(f"Task ID: {task_id}\n{true}\n{pred}")

show_img_from_df(df, 82)

# df_hard[~df_hard["task_id"].isin(excluded_id)].sort_values("accuracy")

# tmp = df[(df["segment"] == "hard") & (df["task_id"] == 4960)]
# tmp
# idx = 2599
# task_id = tmp[tmp.index == idx]["task_id"].item()
# x1, y1, x2, y2 = tmp[tmp.index == idx]["box"].item()
# path = tmp[tmp.index == idx]["path"].item()
# img = Image.open(path)
# img = ImageOps.exif_transpose(img)
# viz_utils.crop_image(img, [(x1, y1, x2, y2)])[0]
