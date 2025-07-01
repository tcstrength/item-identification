"""
%load_ext autoreload
%autoreload 2
"""
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
from hcmus.utils import data_utils, viz_utils
from hcmus.utils import transform_utils
from hcmus.models.prototype import PrototypicalNetwork
from hcmus.models.prototype import PrototypeTracker

splits = data_utils.get_data_splits()

transform_train, transform_test = transform_utils.get_transforms_v2()
datasets = data_utils.get_image_datasets(splits, transform_train, transform_test)

n_way = 5
n_shot = 10
n_query = 5
n_tasks = 32

train_sampler = TaskSampler(
    dataset=datasets["train"],
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_tasks=n_tasks
)
train_dataloader = DataLoader(
    datasets["train"],
    batch_sampler=train_sampler,
    collate_fn=train_sampler.episodic_collate_fn
)

support_sampler = TaskSampler(
    dataset=datasets["support"],
    n_way=len(datasets["support"].label2idx),
    n_shot=n_shot,
    n_query=0,
    n_tasks=32
)
support_dataloader = DataLoader(
    datasets["support"],
    batch_sampler=support_sampler,
    collate_fn=support_sampler.episodic_collate_fn
)

val_dataloader = DataLoader(
    datasets["val"],
    shuffle=False,
    batch_size=32
)


model = PrototypicalNetwork(freeze_clip=False)
tracker = PrototypeTracker()
epochs = 5
for item in range(epochs):
    for episode in tqdm(train_dataloader, desc="Computing prototypes..."):
        support_images, support_labels, query_images, query_labels, classes = episode
        support_features = model.encode_images(support_images)
        individual_prototypes = model.compute_prototypes(support_features, support_labels, len(classes))
        logits, _ = model.forward(prototypes=individual_prototypes, query_images=query_images)
        # print(f"Individual: {sum(logits.argmax(dim=1) == query_labels) / len(query_labels)}")
        tracker.update(individual_prototypes, classes)
        # logits, _ = model.forward(prototypes=tracker.get_prototypes(classes), query_images=query_images)
        # print(f"Average: {sum(logits.argmax(dim=1) == query_labels) / len(query_labels)}")

items = sorted(tracker.prototypes.keys())
for batch in tqdm(val_dataloader, desc="Evaluating..."):
    query_images, query_labels = batch
    logits, _ = model.forward(prototypes=tracker.get_prototypes(items), query_images=query_images)
    print(sum(logits.argmax(dim=1) == query_labels) / len(query_labels))

# support_labels, classes
# tracker.prototypes[95].shape
# individual_prototypes.shape
# tracker.get_prototypes(classes).shape

# len(tracker.prototypes)
