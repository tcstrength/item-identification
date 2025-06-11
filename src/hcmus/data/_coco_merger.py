import json
import random
import shutil
import yaml
from pathlib import Path
from loguru import logger

class COCODatasetMerger:
    """Merge two COCO datasets with custom weights per split and selective split processing"""

    def __init__(self, dataset1_path, dataset2_path, output_path, split_weights=None):
        self.dataset1_path = Path(dataset1_path)
        self.dataset2_path = Path(dataset2_path)
        self.output_path = Path(output_path)

        # Handle split weights - can be per-split or global
        if isinstance(split_weights, dict):
            self.split_weights = split_weights
        else:
            raise ValueError("split_weights must be a dictionary")

        # Set which splits to merge
        self.splits_to_merge = split_weights.keys()  # Default: merge all splits
        for split in self.splits_to_merge:
            if split not in self.split_weights:
                self.split_weights[split] = {'weight1': 1, 'weight2': 1}

            # Ensure both weight1 and weight2 are specified
            if 'weight1' not in self.split_weights[split]:
                self.split_weights[split]['weight1'] = 1
            if 'weight2' not in self.split_weights[split]:
                self.split_weights[split]['weight2'] = 1

        self.num_classes = 0  # Will be set during merging
        self.merged_categories = []  # Will store merged categories

        logger.info(f"Splits to merge: {self.splits_to_merge}")
        logger.info(f"Split weights: {self.split_weights}")

    def load_coco_annotation(self, ann_path):
        """Load COCO annotation file"""
        with open(ann_path, 'r') as f:
            return json.load(f)

    def merge_datasets(self):
        """Merge two COCO datasets with custom weights for selected splits"""
        logger.info("Starting dataset merging...")

        # First, analyze categories from both datasets to determine number of classes
        self._analyze_categories()

        # Create output directory structure only for splits being merged
        for split in self.splits_to_merge:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)

        # Process each selected split
        for split in self.splits_to_merge:
            self._merge_split(split)

        logger.info("Dataset merging completed!")
        logger.info(f"Processed splits: {self.splits_to_merge}")
        logger.info(f"Total number of classes: {self.num_classes}")
        logger.info(f"Categories: {[cat['name'] for cat in self.merged_categories]}")

        return self.num_classes, self.merged_categories

    def _analyze_categories(self):
        """Analyze categories from both datasets to determine total number of classes"""
        logger.info("Analyzing categories from both datasets...")

        all_categories = []

        # Get categories from both datasets by checking any available split
        for dataset_path in [self.dataset1_path, self.dataset2_path]:
            categories_found = False
            # Check all possible splits, not just the ones being merged, to get complete category list
            for split in self.splits_to_merge:
                ann_path = dataset_path / split / f'annotations_{split}.json'
                if ann_path.exists():
                    ann_data = self.load_coco_annotation(ann_path)
                    if 'categories' in ann_data and ann_data['categories']:
                        all_categories.extend(ann_data['categories'])
                        categories_found = True
                        break

            if not categories_found:
                logger.warning(f"No categories found in dataset: {dataset_path}")

        # Merge categories and remove duplicates
        self.merged_categories = self._merge_categories_list(all_categories)
        self.num_classes = len(self.merged_categories) + 1  # +1 for background class

        logger.info(f"Found {len(self.merged_categories)} unique categories")
        logger.info(f"Total classes (including background): {self.num_classes}")

    def _merge_categories_list(self, all_categories):
        """Merge category lists from multiple sources, removing duplicates"""
        merged_cats = []
        seen_names = set()

        for cat in all_categories:
            if cat['name'] not in seen_names:
                merged_cats.append(cat.copy())
                seen_names.add(cat['name'])

        # Reassign IDs to be sequential starting from 1
        for i, cat in enumerate(merged_cats):
            cat['id'] = i + 1

        return merged_cats

    def _merge_split(self, split):
        """Merge a specific split (train/val/test) with its custom weights"""
        logger.info(f"Merging {split} split...")

        # Get weights for this specific split
        weight1 = self.split_weights[split]['weight1']
        weight2 = self.split_weights[split]['weight2']

        logger.info(f"Using weights for {split}: dataset1={weight1}, dataset2={weight2}")

        # Load annotations
        ann1_path = self.dataset1_path / split / f'annotations_{split}.json'
        ann2_path = self.dataset2_path / split / f'annotations_{split}.json'

        if not ann1_path.exists() and not ann2_path.exists():
            logger.warning(f"Skipping {split} split - no annotation files found")
            return

        # Load available annotations
        ann1 = self.load_coco_annotation(ann1_path) if ann1_path.exists() else {'images': [], 'annotations': []}
        ann2 = self.load_coco_annotation(ann2_path) if ann2_path.exists() else {'images': [], 'annotations': []}

        # Calculate number of samples based on weights
        total_samples1 = len(ann1['images'])
        total_samples2 = len(ann2['images'])

        if total_samples1 == 0 and total_samples2 == 0:
            logger.warning(f"No images found in {split} split")
            return

        # Sample images based on split-specific weights
        if isinstance(weight1, int) and weight1 >= 1:
            # Integer weight means repeat the dataset
            sampled_imgs1 = ann1['images'] * weight1
        elif isinstance(weight1, (int, float)) and 0 <= weight1 < 1:
            # Float weight means sample a fraction
            sampled_imgs1 = random.sample(ann1["images"], int(total_samples1 * weight1))
        else:
            # Default to using all images
            sampled_imgs1 = ann1['images']

        if isinstance(weight2, int) and weight2 >= 1:
            # Integer weight means repeat the dataset
            sampled_imgs2 = ann2['images'] * weight2
        elif isinstance(weight2, (int, float)) and 0 <= weight2 < 1:
            # Float weight means sample a fraction
            sampled_imgs2 = random.sample(ann2["images"], int(total_samples2 * weight2))
        else:
            # Default to using all images
            sampled_imgs2 = ann2['images']

        logger.info(f"Dataset 1 images: {total_samples1} -> {len(sampled_imgs1)} (weight: {weight1})")
        logger.info(f"Dataset 2 images: {total_samples2} -> {len(sampled_imgs2)} (weight: {weight2})")

        # Create merged annotation using pre-analyzed categories
        merged_ann = {
            'info': ann1.get('info', {}) if ann1_path.exists() else ann2.get('info', {}),
            'licenses': ann1.get('licenses', []) if ann1_path.exists() else ann2.get('licenses', []),
            'categories': self.merged_categories,  # Use pre-analyzed categories
            'images': [],
            'annotations': []
        }

        # Create category ID mapping for both datasets
        cat_id_map1 = self._create_category_mapping(ann1.get('categories', []))
        cat_id_map2 = self._create_category_mapping(ann2.get('categories', []))

        # Process dataset 1
        img_id_mapping1 = {}
        ann_id_counter = 1

        for img in sampled_imgs1:
            new_img_id = len(merged_ann['images']) + 1
            img_id_mapping1[img['id']] = new_img_id

            new_img = img.copy()
            new_img['id'] = new_img_id
            merged_ann['images'].append(new_img)

            # Copy image file
            src_img = self.dataset1_path / split / 'images' / img['file_name']
            dst_img = self.output_path / split / 'images' / img['file_name']
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

        # Add annotations for dataset 1
        for ann in ann1.get('annotations', []):
            if ann['image_id'] in img_id_mapping1:
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = img_id_mapping1[ann['image_id']]
                # Map category ID to merged category system
                if ann['category_id'] in cat_id_map1:
                    new_ann['category_id'] = cat_id_map1[ann['category_id']]
                    merged_ann['annotations'].append(new_ann)
                    ann_id_counter += 1

        # Process dataset 2
        img_id_mapping2 = {}

        for img in sampled_imgs2:
            new_img_id = len(merged_ann['images']) + 1
            img_id_mapping2[img['id']] = new_img_id

            new_img = img.copy()
            new_img['id'] = new_img_id
            merged_ann['images'].append(new_img)

            # Copy image file
            src_img = self.dataset2_path / split / 'images' / img['file_name']
            dst_img = self.output_path / split / 'images' / img['file_name']
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

        # Add annotations for dataset 2
        for ann in ann2.get('annotations', []):
            if ann['image_id'] in img_id_mapping2:
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = img_id_mapping2[ann['image_id']]
                # Map category ID to merged category system
                if ann['category_id'] in cat_id_map2:
                    new_ann['category_id'] = cat_id_map2[ann['category_id']]
                    merged_ann['annotations'].append(new_ann)
                    ann_id_counter += 1

        # Save merged annotation
        output_ann_path = self.output_path / split / f'annotations_{split}.json'
        with open(output_ann_path, 'w') as f:
            json.dump(merged_ann, f)

        logger.info(f"{split} split merged: {len(merged_ann['images'])} images, {len(merged_ann['annotations'])} annotations")

    def _create_category_mapping(self, categories):
        """Create mapping from original category IDs to merged category IDs"""
        cat_map = {}
        for cat in categories:
            # Find corresponding category in merged categories
            for merged_cat in self.merged_categories:
                if merged_cat['name'] == cat['name']:
                    cat_map[cat['id']] = merged_cat['id']
                    break
        return cat_map

    def _merge_categories(self, cats1, cats2):
        """Legacy method - kept for compatibility"""
        return self._merge_categories_list(cats1 + cats2)
