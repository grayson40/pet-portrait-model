import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import cv2
from pathlib import Path
import numpy as np


class PetPortraitDataset(Dataset):
    def __init__(self, data_dir, phase="train"):
        self.data_dir = Path(data_dir)
        self.phase = phase

        # Only use part2 directory and categories
        self.part2_dir = self.data_dir / "animal_poses/animalpose_image_part2"
        self.categories = ["cat", "cow", "dog", "horse", "sheep"]

        # Load keypoints data
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

                print(
                    f"Loaded {len(self.images_data)} valid images and {len(self.annotations_data)} annotations"
                )

                # Create image_id to annotation mapping
                self.image_to_annotation = {}
                for ann in self.annotations_data:
                    img_id = ann["image_id"]
                    if img_id not in self.image_to_annotation:
                        self.image_to_annotation[img_id] = []
                    self.image_to_annotation[img_id].append(ann)

        except FileNotFoundError as e:
            print(f"Error loading keypoints.json: {e}")
            self.images_data = {}
            self.annotations_data = []

        # Setup image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load image paths and annotations
        self.samples = self._load_samples()
        print(f"Found {len(self.samples)} valid samples")

    def _find_image_path(self, img_name):
        """Find image in part2 directory categories"""
        for category in self.categories:
            path = self.part2_dir / category / img_name
            if path.exists():
                return path
        return None

    def _load_samples(self):
        samples = []

        # Iterate through the filtered images data
        for img_id, img_name in self.images_data.items():
            img_id = int(img_id)
            img_path = self._find_image_path(img_name)

            if img_path is not None and img_id in self.image_to_annotation:
                annotations = self.image_to_annotation[img_id]
                samples.append(
                    {"path": img_path, "image_id": img_id, "annotations": annotations}
                )

        return samples

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

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform image
        image = cv2.imread(str(sample["path"]))
        if image is None:
            print(f"Failed to load image: {sample['path']}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        # Process annotations
        processed = self._process_keypoints(sample["annotations"])
        if processed is None:
            processed = {
                "is_looking": False,
                "pose_quality": 0.0,
                "bbox": [0, 0, 0, 0],
                "keypoints": np.zeros((20, 3)),
            }

        return {
            "image": image,
            "is_looking": torch.tensor(processed["is_looking"], dtype=torch.float),
            "pose_quality": torch.tensor(processed["pose_quality"], dtype=torch.float),
            "bbox": torch.tensor(processed["bbox"], dtype=torch.float),
            "keypoints": torch.tensor(processed["keypoints"], dtype=torch.float),
        }

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # Test the dataset loader
    print("Initializing dataset...")
    dataset = PetPortraitDataset("data/raw")

    if len(dataset) == 0:
        print("No samples found in dataset!")
    else:
        print(f"\nDataset contains {len(dataset)} samples")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Is looking at camera: {sample['is_looking']}")
        print(f"Pose quality: {sample['pose_quality']}")
        print(f"Bounding box: {sample['bbox']}")
        print(f"Number of keypoints: {sample['keypoints'].shape}")

        # Print category distribution
        image_paths = [str(s["path"]) for s in dataset.samples]
        categories = [p.split("/")[-2] for p in image_paths]
        category_counts = {cat: categories.count(cat) for cat in dataset.categories}
        print("\nImages per category:")
        for cat, count in category_counts.items():
            print(f"{cat}: {count} images")
