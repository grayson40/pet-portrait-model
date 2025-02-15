import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import json
import random


class BalancedPetDataset(Dataset):
    def __init__(self, data_dir, phase="train"):
        self.data_dir = Path(data_dir)
        self.phase = phase

        # Different transforms for training and validation
        if self.phase == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.3,
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3)], p=0.1
                    ),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomErasing(p=0.1),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Animal Pose dataset setup
        self.categories = ["cat", "cow", "dog", "horse", "sheep"]
        self.part2_dir = self.data_dir / "animal_poses/animalpose_image_part2"

        # Load keypoints data for Animal Pose dataset
        try:
            with open(self.data_dir / "animal_poses/keypoints.json", "r") as f:
                data = json.load(f)
                # Filter images that start with our categories
                self.images_data = {
                    k: v
                    for k, v in data["images"].items()
                    if any(
                        v.startswith(prefix)
                        for prefix in ["ca", "co", "do", "ho", "sh"]
                    )
                }

                # Only keep annotations for our filtered images
                valid_image_ids = set(int(k) for k in self.images_data.keys())
                self.annotations_data = [
                    ann
                    for ann in data["annotations"]
                    if ann["image_id"] in valid_image_ids
                ]

                # Create image_id to annotation mapping
                self.image_to_annotation = {}
                for ann in self.annotations_data:
                    img_id = ann["image_id"]
                    if img_id not in self.image_to_annotation:
                        self.image_to_annotation[img_id] = []
                    self.image_to_annotation[img_id].append(ann)

                print(
                    f"Loaded {len(self.images_data)} valid images and {len(self.annotations_data)} annotations"
                )

        except FileNotFoundError as e:
            print(f"Error loading keypoints.json: {e}")
            self.images_data = {}
            self.annotations_data = []
            self.image_to_annotation = {}

        # Load datasets
        self.animal_pose_samples = self._load_animal_pose()
        self.oxford_samples = self._load_oxford()
        self.stanford_samples = self._load_stanford()

        # Balance dataset
        self.samples = self._balance_dataset()

        print(f"\nDataset Statistics ({phase} split):")
        print(f"Animal Pose samples: {len(self.animal_pose_samples)}")
        print(f"Oxford samples: {len(self.oxford_samples)}")
        print(f"Stanford samples: {len(self.stanford_samples)}")
        print(f"Total balanced samples: {len(self.samples)}")

    def _find_image_path(self, img_name):
        """Find image in part2 directory categories"""
        for category in self.categories:
            path = self.part2_dir / category / img_name
            if path.exists():
                return str(path)
        return None

    def _process_keypoints(self, annotations):
        """Process keypoints for a single image"""
        if not annotations:
            return None

        ann = annotations[0]
        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)

        # Extract face keypoints (first 5 points typically include face features)
        face_points = keypoints[:5]

        # Calculate if looking at camera based on keypoint visibility
        visible_face_points = face_points[face_points[:, 2] > 0]
        is_looking = len(visible_face_points) >= 3

        # Calculate pose quality based on number of visible keypoints
        total_points = len(keypoints)
        visible_points = np.sum(keypoints[:, 2] > 0)
        pose_quality = visible_points / total_points

        return {
            "is_looking": is_looking,
            "pose_quality": pose_quality,
            "bbox": ann["bbox"],
            "keypoints": keypoints,
        }

    def _load_animal_pose(self):
        """Load Animal Pose dataset with existing annotations"""
        samples = []

        for img_id, img_name in self.images_data.items():
            img_id = int(img_id)
            img_path = self._find_image_path(img_name)

            if img_path is not None and img_id in self.image_to_annotation:
                annotations = self.image_to_annotation[img_id]
                processed = self._process_keypoints(annotations)
                if processed:
                    samples.append(
                        {
                            "path": img_path,
                            "is_looking": processed["is_looking"],
                            "pose_quality": processed["pose_quality"],
                            "dataset": "animal_pose",
                            "keypoints": processed["keypoints"],
                            "bbox": processed["bbox"],
                        }
                    )

        print(f"Loaded {len(samples)} samples from Animal Pose dataset")
        return samples

    def _load_oxford(self):
        """
        Load Oxford dataset and determine looking/not looking
        """
        oxford_dir = self.data_dir / "oxford_pets"
        samples = []

        # Load annotations
        annot_dir = oxford_dir / "annotations/xmls"
        if not annot_dir.exists():
            return samples

        for xml_file in annot_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image path
                img_name = xml_file.stem + ".jpg"
                img_path = oxford_dir / "images" / img_name

                if not img_path.exists():
                    continue

                # Determine if looking at camera based on pose annotation
                # This is a simplified heuristic - you might want to improve it
                is_looking = "frontal" in root.find(".//pose").text.lower()

                samples.append(
                    {"path": img_path, "is_looking": is_looking, "dataset": "oxford"}
                )

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

        return samples

    def _load_stanford(self):
        """
        Load Stanford dataset and determine looking/not looking
        """
        stanford_dir = self.data_dir / "stanford_dogs"
        samples = []

        # Load annotations
        annot_dir = stanford_dir / "Annotation"
        if not annot_dir.exists():
            print(f"Stanford Dogs annotation directory not found at {annot_dir}")
            return samples

        # First, find all breed directories
        breed_dirs = [d for d in annot_dir.iterdir() if d.is_dir()]

        # Then process XML files within each breed directory
        for breed_dir in breed_dirs:
            # Process all XML files in this breed directory - look for files without extension
            for xml_file in breed_dir.glob(
                "n*[0-9]"
            ):  # Match files like n02085620_7 (no extension)
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # Get folder (breed id) and filename from XML
                    breed_id = root.find("folder").text  # e.g., "02085620"
                    breed_name = root.find(".//name").text  # e.g., "Chihuahua"
                    filename = (
                        xml_file.name
                    )  # Use the actual filename without trying to parse from XML

                    # Construct image path using breed directory name
                    breed_dir_name = breed_dir.name  # e.g., "n02085620-Chihuahua"
                    img_path = (
                        stanford_dir / "Images" / breed_dir_name / f"{filename}.jpg"
                    )

                    if not img_path.exists():
                        print(f"Image not found: {img_path}")
                        continue

                    # Get bounding box
                    bbox = root.find(".//bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    # Calculate aspect ratio of bounding box
                    width = xmax - xmin
                    height = ymax - ymin
                    aspect_ratio = width / height

                    # Use aspect ratio and relative size to determine if looking at camera
                    # Dogs looking at camera typically have wider, more square faces
                    bbox_size = (width * height) / (
                        float(root.find(".//width").text)
                        * float(root.find(".//height").text)
                    )
                    is_looking = (
                        aspect_ratio > 0.8 and aspect_ratio < 1.3 and bbox_size > 0.3
                    )

                    samples.append(
                        {
                            "path": str(img_path),
                            "is_looking": is_looking,
                            "dataset": "stanford",
                            "breed": breed_name,
                            "bbox": [xmin, ymin, xmax, ymax],
                        }
                    )

                except Exception as e:
                    print(f"Error processing {xml_file}: {str(e)}")

        looking_count = sum(1 for s in samples if s["is_looking"])
        print(f"\nLoaded {len(samples)} samples from Stanford Dogs dataset")
        print(f"Looking at camera: {looking_count}")
        print(f"Not looking: {len(samples) - looking_count}")

        return samples

    def _balance_dataset(self):
        """Balance the dataset with additional filtering"""
        all_samples = (
            self.animal_pose_samples + self.oxford_samples + self.stanford_samples
        )

        # Split into looking and not looking
        looking = [s for s in all_samples if s["is_looking"]]
        not_looking = [s for s in all_samples if not s["is_looking"]]

        # Shuffle both lists
        random.shuffle(looking)
        random.shuffle(not_looking)

        # Balance the dataset
        target_size = min(len(looking), len(not_looking))
        balanced_looking = looking[:target_size]
        balanced_not_looking = not_looking[:target_size]

        # Combine and shuffle final dataset
        combined = balanced_looking + balanced_not_looking
        random.shuffle(combined)

        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)

        # Create return dictionary with common fields
        item = {
            "image": image,
            "is_looking": torch.tensor(float(sample["is_looking"])),
            "dataset": sample["dataset"],
        }

        # Add pose quality and keypoints if available (for animal pose dataset)
        if sample["dataset"] == "animal_pose":
            item.update(
                {
                    "pose_quality": torch.tensor(
                        sample["pose_quality"], dtype=torch.float
                    ),
                    "keypoints": torch.tensor(sample["keypoints"], dtype=torch.float),
                    "bbox": torch.tensor(sample["bbox"], dtype=torch.float),
                }
            )
        else:
            # Default values for other datasets
            item.update(
                {
                    "pose_quality": torch.tensor(0.0, dtype=torch.float),
                    "keypoints": torch.zeros((20, 3), dtype=torch.float),
                    "bbox": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float),
                }
            )

        return item


if __name__ == "__main__":
    # Test the balanced dataset
    dataset = BalancedPetDataset("data/raw")

    # Print statistics
    looking_count = sum(1 for s in dataset.samples if s["is_looking"])
    not_looking_count = len(dataset.samples) - looking_count

    print(f"\nFinal Dataset Statistics:")
    print(f"Looking at camera: {looking_count}")
    print(f"Not looking at camera: {not_looking_count}")

    # Test loading
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
