"""
%load_ext autoreload
%autoreload 2
"""
from collections import Counter
from tqdm import tqdm
from hcmus.utils import data_utils, viz_utils
from hcmus.utils import transform_utils
from hcmus.models.search import SearchModel

splits = data_utils.get_data_splits()

for scale in (32, 64, 96, 128, 160):
    transform_train, transform_test = transform_utils.get_transforms_downscale(scale)
    datasets = data_utils.get_image_datasets(splits, transform_train, transform_test)
    dataloaders = data_utils.get_data_loaders(datasets, {"train": True})

    # img, label = datasets.get("test")[1]
    # print(label)
    # viz_utils.plot_image(img.detach().numpy().transpose(1, 2, 0))

    ensemble_epochs = 3
    top_k = 5
    # model = SearchModel()
    model = SearchModel(mlflow_logged_model="runs:/16c530d050a24b7095a3bd2585f2c789/model")

    for _ in range(ensemble_epochs):
        for batch in tqdm(dataloaders["train"], "Ensembling..."):
            images, labels = batch
            model.create_embeddings(images).shape
            model.add_support_set(images, labels)

    accuracy = 0
    misclassified = []
    total = len(datasets["val"])
    for img, label in tqdm(datasets["val"], "Evaluating..."):
        result = model.forward(img, k=top_k)[0]
        counter = Counter(result.get("L"))
        pred = counter.most_common(1)[0][0]
        accuracy += pred == label
        if pred != label:
            misclassified.append((img, label, result))

    print("Val", accuracy / total)

    accuracy = 0
    misclassified = []
    total = len(datasets["test"])
    for img, label in tqdm(datasets["test"], "Evaluating..."):
        result = model.forward(img, k=top_k)[0]
        counter = Counter(result.get("L"))
        pred = counter.most_common(1)[0][0]
        accuracy += pred == label
        if pred != label:
            misclassified.append((img, label, result))

    print("Test", accuracy / total)

