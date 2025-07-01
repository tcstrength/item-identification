"""
%load_ext autoreload
%autoreload 2
"""
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from loguru import logger
from hcmus.utils import data_utils, viz_utils
from hcmus.utils import transform_utils
from hcmus.models.search import SearchModel

suffix = datetime.now().strftime("%H:%M:%S")
logger.add(f"logs/2025-06-29-clip-search-visual-only-{suffix}.log")
splits = data_utils.get_data_splits()

for scale in (96, 128, 224):
    logger.info(f"Downscale: {scale}")

    transform_train, transform_test = transform_utils.get_transforms_downscale_random_v2(scale)
    datasets = data_utils.get_image_datasets_v2(splits, transform_train, transform_test, random_margin=0)
    dataloaders = data_utils.get_data_loaders_v2(datasets, {"train": True})

    # img, label = datasets.get("test")[1]
    # print(label)
    # viz_utils.plot_image(img.detach().numpy().transpose(1, 2, 0))

    ensemble_epochs = 3
    top_k = 1
    logged_id = "runs:/e561bde201e24ed5a99f13826a0ad007/model"
    logger.info(f"MLFlow Run ID: {logged_id}")
    logger.info(f"Ensemble epochs: {ensemble_epochs}")
    logger.info(f"Majority voting from top K: {top_k}")
    # model = SearchModel()
    model = SearchModel(mlflow_logged_model=logged_id)

    for _ in range(ensemble_epochs):
        for batch in tqdm(dataloaders["train"], "Ensembling..."):
            images, labels = batch
            model.create_embeddings(images).shape
            model.add_support_set(images, labels)

    accuracy = 0
    misclassified = []
    total = len(datasets["val"])
    for img, label, _ in tqdm(datasets["val"], "Evaluating..."):
        result = model.forward(img, k=top_k)[0]
        counter = Counter(result.get("L"))
        pred = counter.most_common(1)[0][0]
        accuracy += pred == label
        if pred != label:
            misclassified.append((img, label, result))

    logger.info(f"Val {accuracy / total}")

    accuracy = 0
    misclassified = []
    total = len(datasets["test"])
    for img, label, _ in tqdm(datasets["test"], "Evaluating..."):
        result = model.forward(img, k=top_k)[0]
        counter = Counter(result.get("L"))
        pred = counter.most_common(1)[0][0]
        accuracy += pred == label
        if pred != label:
            misclassified.append((img, label, result))

    logger.info(f"Test {accuracy / total}")

