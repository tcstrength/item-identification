import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
from hcmus.lbs._label_studio_models import DatasetItem, DatasetTarget, TargetBox, TargetLabel


class COCOExporter:
    """
    Exports a list of DatasetItem objects to COCO format.

    COCO format structure:
    - info: Dataset information
    - licenses: License information
    - images: List of image metadata
    - annotations: List of object annotations
    - categories: List of category definitions
    """

    def __init__(self,
                 dataset_name: str = "Custom Dataset",
                 description: str = "Custom dataset exported to COCO format",
                 version: str = "1.0",
                 year: int = None,
                 contributor: str = "Auto-generated",
                 url: str = "",
                 copy_images: bool = True):
        """
        Initialize COCO exporter.

        Args:
            dataset_name: Name of the dataset
            description: Description of the dataset
            version: Version string
            year: Year of creation
            contributor: Dataset contributor
            url: Dataset URL
            copy_images: Whether to copy images to output directory
        """
        self.dataset_name = dataset_name
        self.description = description
        self.version = version
        self.year = year or datetime.now().year
        self.contributor = contributor
        self.url = url
        self.copy_images = copy_images

        # COCO format structure
        self.coco_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Tracking variables
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.categories_map = {}  # label_str -> category_id
        self.image_filename_to_id = {}  # filename -> image_id

    def _create_info_section(self) -> Dict[str, Any]:
        """Create the info section of COCO format."""
        return {
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "year": self.year,
            "contributor": self.contributor,
            "date_created": datetime.now().isoformat()
        }

    def _create_default_license(self) -> List[Dict[str, Any]]:
        """Create default license information."""
        return [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ]

    def _extract_categories(self, dataset_items: List[DatasetItem]) -> None:
        """Extract unique categories from dataset items."""
        unique_labels = set()

        for item in dataset_items:
            for label in item.target.labels:
                unique_labels.add(label.label_str)

        # Create categories with sequential IDs starting from 1
        for idx, label_str in enumerate(sorted(unique_labels), start=1):
            category = {
                "id": idx,
                "name": label_str,
                "supercategory": "object"  # Default supercategory
            }
            self.coco_data["categories"].append(category)
            self.categories_map[label_str] = idx

    def _get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Extract image information from image file."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            # Use default dimensions if image can't be read
            width, height = 640, 480

        filename = os.path.basename(image_path)

        return {
            "id": self.image_id_counter,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().isoformat()
        }

    def _convert_box_to_coco_format(self, box: TargetBox) -> List[float]:
        """
        Convert TargetBox (xmin, ymin, xmax, ymax) to COCO format [x, y, width, height].

        Args:
            box: TargetBox with xmin, ymin, xmax, ymax

        Returns:
            List in COCO format: [x, y, width, height]
        """
        x = float(box.xmin)
        y = float(box.ymin)
        width = float(box.xmax - box.xmin)
        height = float(box.ymax - box.ymin)

        return [x, y, width, height]

    def _calculate_area(self, box: TargetBox) -> float:
        """Calculate area of bounding box."""
        return float((box.xmax - box.xmin) * (box.ymax - box.ymin))

    def _create_annotation(self,
                          box: TargetBox,
                          label: TargetLabel,
                          image_id: int) -> Dict[str, Any]:
        """Create a single annotation in COCO format."""
        bbox_coco = self._convert_box_to_coco_format(box)
        area = self._calculate_area(box)
        category_id = self.categories_map[label.label_str]

        annotation = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],  # Empty for bounding box only
            "area": area,
            "bbox": bbox_coco,
            "iscrowd": 0
        }

        self.annotation_id_counter += 1
        return annotation

    def _copy_image_if_needed(self, source_path: str, output_dir: str) -> str:
        """Copy image to output directory if copy_images is True."""
        if not self.copy_images:
            return source_path

        # Create images subdirectory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        filename = os.path.basename(source_path)
        destination_path = os.path.join(images_dir, filename)

        # Copy file if it doesn't exist or if source is newer
        if not os.path.exists(destination_path) or \
           os.path.getmtime(source_path) > os.path.getmtime(destination_path):
            try:
                shutil.copy2(source_path, destination_path)
            except Exception as e:
                print(f"Warning: Could not copy image {source_path}: {e}")

        return destination_path

    def export(self,
               dataset_items: List[DatasetItem],
               output_path: str,
               split_name: str = "train") -> Dict[str, Any]:
        """
        Export dataset items to COCO format.

        Args:
            dataset_items: List of DatasetItem objects to export
            output_path: Directory where to save the COCO files
            split_name: Name of the split (train, val, test)

        Returns:
            Dictionary containing export statistics
        """
        if not dataset_items:
            raise ValueError("Dataset items list is empty")

        # Reset counters for new export
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.categories_map = {}
        self.image_filename_to_id = {}

        # Initialize COCO data structure
        self.coco_data = {
            "info": self._create_info_section(),
            "licenses": self._create_default_license(),
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Extract categories first
        self._extract_categories(dataset_items)

        print(f"Found {len(self.coco_data['categories'])} categories:")
        for cat in self.coco_data['categories']:
            print(f"  - {cat['name']} (ID: {cat['id']})")

        # Process each dataset item
        print(f"\nProcessing {len(dataset_items)} dataset items...")

        for idx, item in enumerate(dataset_items):
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(dataset_items)} items...")

            # Handle image
            if not os.path.exists(item.image):
                print(f"Warning: Image not found: {item.image}")
                continue

            # Copy image if needed
            final_image_path = self._copy_image_if_needed(item.image, output_path)

            # Create image info
            image_info = self._get_image_info(item.image)
            filename = os.path.basename(item.image)

            # Update filename if image was copied
            if self.copy_images:
                image_info["file_name"] = os.path.basename(final_image_path)

            self.coco_data["images"].append(image_info)
            self.image_filename_to_id[filename] = self.image_id_counter

            # Validate that boxes and labels have the same length
            if len(item.target.boxes) != len(item.target.labels):
                print(f"Warning: Mismatch in boxes ({len(item.target.boxes)}) and labels ({len(item.target.labels)}) for image {filename}")
                min_length = min(len(item.target.boxes), len(item.target.labels))
                boxes = item.target.boxes[:min_length]
                labels = item.target.labels[:min_length]
            else:
                boxes = item.target.boxes
                labels = item.target.labels

            # Create annotations for this image
            for box, label in zip(boxes, labels):
                # Validate box coordinates
                if box.xmax <= box.xmin or box.ymax <= box.ymin:
                    print(f"Warning: Invalid box coordinates in image {filename}: {box}")
                    continue

                # Validate label
                if label.label_str not in self.categories_map:
                    print(f"Warning: Unknown label '{label.label_str}' in image {filename}")
                    continue

                annotation = self._create_annotation(box, label, self.image_id_counter)
                self.coco_data["annotations"].append(annotation)

            self.image_id_counter += 1

        # Save COCO format JSON
        output_file = os.path.join(output_path, f"annotations_{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

        # Create statistics
        stats = {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "total_categories": len(self.coco_data["categories"]),
            "categories": {cat["name"]: cat["id"] for cat in self.coco_data["categories"]},
            "output_file": output_file,
            "images_copied": self.copy_images
        }

        print(f"\nExport completed!")
        print(f"Images: {stats['total_images']}")
        print(f"Annotations: {stats['total_annotations']}")
        print(f"Categories: {stats['total_categories']}")
        print(f"Output file: {output_file}")

        return stats

    def validate_export(self, output_path: str, split_name: str = "train") -> Dict[str, Any]:
        """
        Validate the exported COCO format file.

        Args:
            output_path: Directory containing the exported files
            split_name: Name of the split to validate

        Returns:
            Dictionary containing validation results
        """
        annotation_file = os.path.join(output_path, f"annotations_{split_name}.json")

        if not os.path.exists(annotation_file):
            return {"valid": False, "error": "Annotation file not found"}

        try:
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)

            # Basic validation
            required_keys = ["info", "licenses", "images", "annotations", "categories"]
            missing_keys = [key for key in required_keys if key not in coco_data]

            if missing_keys:
                return {"valid": False, "error": f"Missing keys: {missing_keys}"}

            # Validate structure
            validation_results = {
                "valid": True,
                "images_count": len(coco_data["images"]),
                "annotations_count": len(coco_data["annotations"]),
                "categories_count": len(coco_data["categories"]),
                "missing_images": []
            }

            # Check if image files exist (if images were copied)
            if self.copy_images:
                images_dir = os.path.join(output_path, "images")
                for img_info in coco_data["images"]:
                    img_path = os.path.join(images_dir, img_info["file_name"])
                    if not os.path.exists(img_path):
                        validation_results["missing_images"].append(img_info["file_name"])

            return validation_results

        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"JSON decode error: {e}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}
