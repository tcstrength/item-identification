import os
import sys
import json
import argparse
from pathlib import Path
import tarfile
import shutil
from datetime import datetime

try:
    import requests
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install requests pandas pillow tqdm")
    sys.exit(1)


class SKU110KDownloader:
    def __init__(self, output_dir: str = "./sku110k_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # URLs for dataset download
        self.dataset_urls = [
            # "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz",
            # "https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd"
            "https://github.com/eg4000/SKU110K_CVPR19"
            # Backup URL (Google Drive requires special handling)
        ]

        self.dataset_path = self.output_dir / "SKU110K_fixed.tar.gz"
        self.extracted_path = self.output_dir / "SKU110K_fixed"

    def download_dataset(self) -> bool:
        """Download the SKU110K dataset"""
        if self.dataset_path.exists():
            print(f"Dataset already exists at {self.dataset_path}")
            return True

        print("Downloading SKU110K dataset...")

        for url in self.dataset_urls:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(self.dataset_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                print(f"Dataset downloaded successfully to {self.dataset_path}")
                return True

            except requests.RequestException as e:
                print(f"Failed to download from {url}: {e}")
                continue

        print("Failed to download dataset from all sources")
        return False

    def extract_dataset(self) -> bool:
        """Extract the downloaded tar.gz file"""
        if self.extracted_path.exists():
            print(f"Dataset already extracted at {self.extracted_path}")
            return True

        if not self.dataset_path.exists():
            print("Dataset file not found. Please download first.")
            return False

        print("Extracting dataset...")
        try:
            with tarfile.open(self.dataset_path, 'r:gz') as tar:
                tar.extractall(self.output_dir)
            print(f"Dataset extracted to {self.extracted_path}")
            return True
        except Exception as e:
            print(f"Failed to extract dataset: {e}")
            return False


class COCOConverter:
    def __init__(self, sku110k_path: Path, output_path: Path):
        self.sku110k_path = Path(sku110k_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "SKU110K Dataset converted to COCO format",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "SKU110K Dataset Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Academic Use Only",
                    "url": "https://github.com/eg4000/SKU110K_CVPR19"
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "supercategory": "retail_item"
                }
            ],
            "images": [],
            "annotations": []
        }

        self.image_id = 1
        self.annotation_id = 1

    def convert_split(self, split_name: str) -> bool:
        """Convert a specific split (train/val/test) to COCO format"""
        print(f"Converting {split_name} split...")

        # Paths for this split
        images_dir = self.sku110k_path / "images"
        annotations_file = self.sku110k_path / "annotations" / f"annotations_{split_name}.csv"

        if not images_dir.exists() or not annotations_file.exists():
            print(f"Missing files for {split_name} split")
            print(f"Images dir: {images_dir.exists()}")
            print(f"Annotations file: {annotations_file.exists()}")
            return False

        # Read annotations CSV
        try:
            df = pd.read_csv(annotations_file, header=None, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'label', 'w', 'h'])
            print(f"Loaded {len(df)} annotations from {annotations_file}")
        except Exception as e:
            print(f"Failed to read annotations file: {e}")
            return False

        # Create output directories
        split_output_dir = self.output_path / split_name
        split_images_dir = split_output_dir / "images"
        split_images_dir.mkdir(parents=True, exist_ok=True)

        # Reset COCO format for this split
        coco_data = self.coco_format.copy()
        coco_data["images"] = []
        coco_data["annotations"] = []

        # Process images and annotations
        image_files = {}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            image_name = row['image_name'] if 'image_name' in row else row['filename']
            image_path = images_dir.joinpath(str(image_name))

            if not image_path.exists():
                continue

            # Add image info (only once per image)
            if image_name not in image_files:
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size

                    image_info = {
                        "id": self.image_id,
                        "file_name": image_name,
                        "width": width,
                        "height": height
                    }
                    coco_data["images"].append(image_info)
                    image_files[image_name] = self.image_id

                    # Copy image to output directory
                    shutil.copy2(image_path, split_images_dir.joinpath(image_name))

                    self.image_id += 1

                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
                    continue

            # Add annotation
            try:
                # Extract bounding box coordinates
                # SKU110K format: x1, y1, x2, y2
                x1 = float(row['x1']) if 'x1' in row else float(row['left'])
                y1 = float(row['y1']) if 'y1' in row else float(row['top'])
                x2 = float(row['x2']) if 'x2' in row else float(row['right'])
                y2 = float(row['y2']) if 'y2' in row else float(row['bottom'])

                # Convert to COCO format: x, y, width, height
                width = x2 - x1
                height = y2 - y1
                area = width * height

                print()

                if width <= 0 or height <= 0:
                    continue

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_files[image_name],
                    "category_id": 1,  # Single category for all objects
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                self.annotation_id += 1

            except Exception as e:
                print(f"Error processing annotation in row {idx}: {e}")
                continue

        # Save COCO annotations
        annotations_output_path = split_output_dir / f"annotations_{split_name}.json"
        with open(annotations_output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"Converted {split_name}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        print(f"Saved to {annotations_output_path}")

        return True

    def convert_all_splits(self) -> bool:
        """Convert all available splits"""
        splits = ['train', 'val', 'test']
        success = True

        for split in splits:
            if not self.convert_split(split):
                success = False
                print(f"Failed to convert {split} split")

        return success


def download(output_dir: str):
    downloader = SKU110KDownloader(output_dir + "/tmp")
    downloader.download_dataset()
    downloader.extract_dataset()
    print(downloader.extracted_path)
    converter = COCOConverter(downloader.extracted_path, output_dir)
    converter.convert_all_splits()
    # if downloader.dataset_path.exists():
    #     downloader.dataset_path.unlink()
    # if downloader.extracted_path.exists():
    #     shutil.rmtree(downloader.extracted_path)
