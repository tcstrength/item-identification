import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
from pathlib import Path
from typing import List
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from hcmus.core import appconfig
from hcmus.lbs import LabelStudioConnector
from hcmus.utils import data_utils

def load_model():
    """Load pre-trained ResNet-50 model and remove the final classification layer"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final fully connected layer to get feature embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for ResNet-50 input"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_features(model, image_tensor):
    """Extract feature vector from image using ResNet-50"""
    with torch.no_grad():
        features = model(image_tensor)
        # Flatten the features (remove spatial dimensions)
        features = features.view(features.size(0), -1)
    return features

def cosine_similarity(feat1, feat2):
    """Compute cosine similarity between two feature vectors"""
    similarity = F.cosine_similarity(feat1, feat2, dim=1)
    return similarity.item()

def compare_images(model, image_path1, image_path2):
    """
    Compare similarity between two images using ResNet-50 features

    Args:
        image_path1 (str): Path to first image
        image_path2 (str): Path to second image

    Returns:
        float: Cosine similarity score between -1 and 1
               (1 = identical, 0 = orthogonal, -1 = opposite)
    """

    # Preprocess images
    img1_tensor = preprocess_image(image_path1)
    img2_tensor = preprocess_image(image_path2)

    # Extract features
    features1 = extract_features(model, img1_tensor)
    features2 = extract_features(model, img2_tensor)

    # Compute cosine similarity
    similarity = cosine_similarity(features1, features2)

    return similarity

def list_all_images(folder_path: str) -> List[str]:
    image_files = []
    extensions = ["jpg", "png", "JPG", "PNG", "jpeg", "JPEG"]
    for ext in extensions:
        image_files.extend(glob.glob(f"{folder_path}/**/*.{ext}", recursive=True))

    return image_files

def copy_high_quality_images():
    input_folder = "/Users/keith/Downloads/Shelves"
    output_folder = "/Volumes/Cucumber/Docker/Services/label-studio/data/media/upload/7"
    input_images = list_all_images(input_folder)
    output_images = list_all_images(output_folder)
    model = load_model()

    for out in output_images:
        out_filename = Path(out).name
        w, h = Image.open(str(out)).size
        if w > 1200 and h > 1200:
            continue

        for inp in input_images:
            inp_filename = Path(inp).name
            if inp_filename in out_filename:
                similarity_score = compare_images(model, inp, out)
                if similarity_score < 0.95:
                    print(out, inp)
                    print(f" . Cosine similarity: {similarity_score:.4f}")
                else:
                    print(f"Copy {inp} to {out}")
                    os.system(f"cp {inp} {out}")

def search_bad_quality_images(split_name: str):
    # split_name="val"
    lsb_connector = LabelStudioConnector(
        url=appconfig.LABEL_STUDIO_URL,
        api_key=appconfig.LABEL_STUDIO_API_KEY,
        project_id=appconfig.LABEL_STUDIO_PROJECT_MAPPING[split_name],
        temp_dir=appconfig.LABEL_STUDIO_TEMP_DIR
    )

    tasks = lsb_connector.get_tasks()

    labels = lsb_connector.extract_labels(tasks)

    dataset = lsb_connector.download_dataset(tasks, labels)
    # dataset = lsb_connector.transform_labels(dataset, labels, data_utils.VALID_LABELS)
    for item in dataset:
        path = item.get("image")
        task = item.get("task")
        w, h = Image.open(path).size
        if w <= 1200 or h <= 1200:
            print(task.id, path, (w, h))

search_bad_quality_images("test")
