import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random
import matplotlib.pyplot as plt
import json
import numpy as np


class BalancedPetDataset(Dataset):
    def __init__(self, data_dir, phase="train"):
        self.data_dir = Path(data_dir)
        self.phase = phase

        # Simpler transforms
        if self.phase == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((192, 192)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((192, 192)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Load all datasets
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

    def _is_looking_oxford(self, root):
        # Get all available information
        pose = root.find(".//pose").text.lower()

        # More detailed pose analysis
        is_looking = any(
            [
                "frontal" in pose,
                "front" in pose,
                not any(angle in pose for angle in ["left", "right", "back"]),
            ]
        )

        return is_looking

    def _load_oxford(self):
        """Load Oxford dataset with scaled coordinates"""
        oxford_dir = self.data_dir / "oxford_pets"
        samples = []

        annot_dir = oxford_dir / "annotations/xmls"
        if not annot_dir.exists():
            return samples

        for xml_file in annot_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get original dimensions
                orig_width = float(root.find(".//width").text)
                orig_height = float(root.find(".//height").text)

                # Calculate scaling factors
                width_scale = 192.0 / orig_width
                height_scale = 192.0 / orig_height

                img_name = xml_file.stem + ".jpg"
                img_path = oxford_dir / "images" / img_name

                if not img_path.exists():
                    continue

                # Check if the image is looking at the camera
                is_looking = self._is_looking_oxford(root)

                # Add scaled bbox if available
                bbox = root.find(".//bndbox")
                if bbox is not None:
                    xmin = float(bbox.find("xmin").text) * width_scale
                    ymin = float(bbox.find("ymin").text) * height_scale
                    xmax = float(bbox.find("xmax").text) * width_scale
                    ymax = float(bbox.find("ymax").text) * height_scale
                else:
                    xmin = ymin = xmax = ymax = 0

                samples.append(
                    {
                        "path": str(img_path),
                        "is_looking": is_looking,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "dataset": "oxford",
                    }
                )

            except Exception as e:
                continue

        return samples

    def _is_looking_stanford(self, bbox, image_size):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        bbox_area = width * height
        image_area = image_size[0] * image_size[1]

        # Better heuristics for "looking at camera":
        face_ratio = width / height
        relative_size = bbox_area / image_area

        # A dog is likely looking at camera when:
        # 1. Face is relatively square (not profile view)
        # 2. Bbox is reasonably sized (not too small/large)
        # 3. Bbox is in upper portion of image
        is_looking = (
            0.7 < face_ratio < 1.4  # More permissive ratio
            and 0.15 < relative_size < 0.8  # Size constraints
            and ymin < image_size[1] * 0.7  # Head position check
        )

        return is_looking

    def _load_stanford(self):
        """Load Stanford dataset with scaled coordinates"""
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
            # Process all XML files in this breed directory
            for xml_file in breed_dir.glob("n*"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # Get original dimensions
                    orig_width = float(root.find(".//width").text)
                    orig_height = float(root.find(".//height").text)

                    # Calculate scaling factors
                    width_scale = 192.0 / orig_width
                    height_scale = 192.0 / orig_height

                    # Construct image path using breed directory name
                    breed_dir_name = breed_dir.name
                    img_path = (
                        stanford_dir
                        / "Images"
                        / breed_dir_name
                        / f"{xml_file.name}.jpg"
                    )

                    if not img_path.exists():
                        # print(f"Image not found: {img_path}")  # Uncomment for debugging
                        continue

                    # Get and scale bbox coordinates
                    bbox = root.find(".//bndbox")
                    if bbox is not None:
                        xmin = float(bbox.find("xmin").text) * width_scale
                        ymin = float(bbox.find("ymin").text) * height_scale
                        xmax = float(bbox.find("xmax").text) * width_scale
                        ymax = float(bbox.find("ymax").text) * height_scale

                        # Calculate if looking using original dimensions
                        is_looking = self._is_looking_stanford(
                            [
                                xmin / width_scale,
                                ymin / height_scale,
                                xmax / width_scale,
                                ymax / height_scale,
                            ],
                            (orig_width, orig_height),
                        )

                        samples.append(
                            {
                                "path": str(img_path),
                                "is_looking": is_looking,
                                "bbox": [xmin, ymin, xmax, ymax],
                                "dataset": "stanford",
                            }
                        )

                except Exception as e:
                    print(f"Error processing {xml_file}: {str(e)}")
                    continue

        print(f"Loaded {len(samples)} samples from Stanford Dogs dataset")
        looking_count = sum(1 for s in samples if s["is_looking"])
        print(f"Looking: {looking_count}, Not looking: {len(samples) - looking_count}")

        return samples

    def _load_animal_pose(self):
        """Load Animal Pose dataset with keypoint-based looking detection"""
        try:
            keypoints_path = self.data_dir / "animal_poses/keypoints.json"
            if not keypoints_path.exists():
                print(f"Error: Keypoints file not found at {keypoints_path}")
                return []

            # Category mapping
            category_map = {
                "ca": "cat",
                "do": "dog",
                "ho": "horse",
                "sh": "sheep",
                "co": "cow",
            }

            with open(keypoints_path, "r") as f:
                data = json.load(f)

            samples = []
            categories_count = {cat: 0 for cat in category_map.values()}

            # Images are directly mapped to filenames
            images_data = data["images"]

            for annotation in data["annotations"]:
                img_id = str(
                    annotation["image_id"]
                )  # Convert to string to match images dict
                if img_id not in images_data:
                    continue

                img_name = images_data[img_id]

                # Get category prefix
                prefix = img_name[:2]
                if prefix not in category_map:
                    continue

                category = category_map[prefix]
                img_path = (
                    self.data_dir
                    / "animal_poses/animalpose_image_part2"
                    / category
                    / img_name
                )
                if not img_path.exists():
                    continue

                # Process keypoints
                keypoints = np.array(annotation["keypoints"], dtype=np.float32).reshape(
                    -1, 3
                )

                # Scale keypoints to 192x192
                orig_width = orig_height = 192  # Since we're resizing to 192x192

                # Scale x, y coordinates but keep visibility flag
                scaled_keypoints = keypoints.copy()
                scaled_keypoints[:, 0] *= 192.0 / orig_width
                scaled_keypoints[:, 1] *= 192.0 / orig_height

                # Scale bbox coordinates
                bbox = annotation["bbox"]
                scaled_bbox = [
                    bbox[0] * (192.0 / orig_width),
                    bbox[1] * (192.0 / orig_height),
                    (bbox[0] + bbox[2]) * (192.0 / orig_width),
                    (bbox[1] + bbox[3]) * (192.0 / orig_height),
                ]

                # Use keypoint-based looking detection
                is_looking = self._is_looking_keypoints(scaled_keypoints)

                samples.append(
                    {
                        "path": str(img_path),
                        "is_looking": is_looking,
                        "bbox": scaled_bbox,
                        "keypoints": scaled_keypoints,
                        "dataset": "animal_pose",
                    }
                )
                categories_count[category] += 1

            # Print statistics
            print("\nAnimal Pose Dataset Statistics:")
            print(f"Total samples: {len(samples)}")
            print(f"Looking at camera: {sum(1 for s in samples if s['is_looking'])}")
            print("\nSamples per category:")
            for cat, count in categories_count.items():
                print(f"{cat}: {count}")

            return samples
        except Exception as e:
            print(f"Error loading Animal Pose dataset: {str(e)}")
            return []

    def _balance_dataset(self):
        """Simple dataset balancing with error checking"""
        # Include animal pose samples in all_samples
        all_samples = (
            self.oxford_samples + self.stanford_samples + self.animal_pose_samples
        )

        if not all_samples:
            print("Warning: No samples loaded from any dataset!")
            return []

        looking = [s for s in all_samples if s["is_looking"]]
        not_looking = [s for s in all_samples if not s["is_looking"]]

        print(f"\nBefore balancing:")
        print(f"Looking samples: {len(looking)}")
        print(f"Not looking samples: {len(not_looking)}")

        # Balance the classes
        target_size = min(len(looking), len(not_looking))
        if target_size == 0:
            print("Warning: One or both classes have no samples!")
            return all_samples

        balanced_looking = random.sample(looking, target_size)
        balanced_not_looking = random.sample(not_looking, target_size)

        # Combine and shuffle
        combined = balanced_looking + balanced_not_looking
        random.shuffle(combined)

        print(f"After balancing: {len(combined)} total samples")
        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "is_looking": torch.tensor(float(sample["is_looking"])),
            "bbox": torch.tensor(sample["bbox"]),
            "dataset": sample["dataset"],
        }

    def _is_looking_keypoints(self, keypoints):
        """Determine if animal is looking at camera using keypoints"""
        # First 5 keypoints are face points
        face_keypoints = keypoints[:5]
        visible_points = face_keypoints[face_keypoints[:, 2] > 0]

        if len(visible_points) >= 3:
            # Get face dimensions
            face_width = np.max(visible_points[:, 0]) - np.min(visible_points[:, 0])
            face_height = np.max(visible_points[:, 1]) - np.min(visible_points[:, 1])

            # Calculate face metrics
            face_ratio = face_width / (face_height + 1e-6)

            # Animal is likely looking at camera if:
            # 1. Face ratio is roughly square (not too wide/narrow)
            # 2. We have enough visible keypoints
            # 3. Face points form a reasonable shape
            is_looking = (
                0.7 < face_ratio < 1.4  # Face shape check
                and face_height > 10  # Minimum face size
            )
            return is_looking
        return False

    def _is_looking_at_camera(self, image, bbox, keypoints=None, dataset_type="oxford"):
        """Unified looking detection across datasets"""
        if dataset_type == "animal_pose" and keypoints is not None:
            return self._is_looking_keypoints(keypoints)

        # For Oxford/Stanford, use bbox-based detection
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        # Calculate face metrics
        face_ratio = width / (height + 1e-6)
        relative_size = (width * height) / (192 * 192)

        # Combined heuristics for bbox-based detection
        is_looking = all(
            [
                0.7 < face_ratio < 1.4,  # Face shape check
                0.15 < relative_size < 0.8,  # Size check
                ymin < 192 * 0.7,  # Position check
                width > 20,  # Minimum width
                height > 20,  # Minimum height
            ]
        )

        return is_looking

    def visualize_sample(self, idx):
        """Visualize a sample with its annotations"""
        sample = self.samples[idx]

        # Load and resize image
        image = Image.open(sample["path"]).convert("RGB")
        image = image.resize((192, 192))
        draw = ImageDraw.Draw(image)

        # Draw bounding box
        bbox = sample["bbox"]
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2)

        # Draw keypoints if available
        detection_method = "BBox-based"
        if sample["dataset"] == "animal_pose" and "keypoints" in sample:
            keypoints = sample["keypoints"]
            detection_method = "Keypoint-based"
            for i, kp in enumerate(keypoints):
                if kp[2] > 0:  # If keypoint is visible
                    x, y = kp[0], kp[1]
                    color = "yellow" if i < 5 else "blue"
                    draw.ellipse(
                        [(x - 2, y - 2), (x + 2, y + 2)], fill=color, outline="red"
                    )

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(
            f"Dataset: {sample['dataset']}\n"
            f"Looking: {sample['is_looking']}\n"
            f"Detection: {detection_method}"
        )
        plt.axis("off")
        plt.show()

    # Add this method to your BalancedPetDataset class
    def verify_annotations(self, num_samples=5):
        """Interactive tool to verify and visualize annotations across datasets"""
        incorrect_labels = []

        # Count of samples per dataset
        dataset_counts = {"animal_pose": 0, "oxford": 0, "stanford": 0}

        # Get samples by dataset
        for sample in self.samples:
            dataset_counts[sample["dataset"]] += 1

        print("\nCurrent Dataset Distribution:")
        for dataset, count in dataset_counts.items():
            print(f"{dataset}: {count} samples")

        while True:
            idx = random.randint(0, len(self.samples) - 1)
            sample = self.samples[idx]

            # Load and show image
            image = Image.open(sample["path"]).convert("RGB")
            image = image.resize((192, 192))
            draw = ImageDraw.Draw(image)

            # Draw bounding box
            bbox = sample["bbox"]
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2
            )

            # Draw keypoints if available
            if sample["dataset"] == "animal_pose" and "keypoints" in sample:
                keypoints = sample["keypoints"]
                # Draw first 5 keypoints (face points) in different color
                for i, kp in enumerate(keypoints):
                    if kp[2] > 0:  # If keypoint is visible
                        x, y = kp[0], kp[1]
                        color = "yellow" if i < 5 else "blue"
                        draw.ellipse(
                            [(x - 2, y - 2), (x + 2, y + 2)], fill=color, outline="red"
                        )

            # Calculate aspect ratio for bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height if height != 0 else 0

            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.title(
                f"Dataset: {sample['dataset']}\n"
                f"Current Label: Looking={sample['is_looking']}\n"
                f"Aspect Ratio: {aspect_ratio:.2f}"
            )
            plt.axis("off")
            plt.show()

            response = input(
                "\nOptions:\n"
                "y: Label is correct\n"
                "n: Label is incorrect\n"
                "q: Quit verification\n"
                "Choice: "
            ).lower()

            if response == "q":
                break
            elif response == "n":
                incorrect_labels.append(
                    {
                        "idx": idx,
                        "dataset": sample["dataset"],
                        "path": sample["path"],
                        "current_label": sample["is_looking"],
                    }
                )

            print("\nVerification Summary:")
            print(f"Samples checked: {len(incorrect_labels)}")
            print("Incorrect labels found:", len(incorrect_labels))

        return incorrect_labels


if __name__ == "__main__":
    # Test the dataset
    dataset = BalancedPetDataset("data/raw")

    # Print statistics
    looking_count = sum(1 for s in dataset.samples if s["is_looking"])
    not_looking_count = len(dataset.samples) - looking_count

    print(f"\nFinal Dataset Statistics:")
    print(f"Looking at camera: {looking_count}")
    print(f"Not looking at camera: {not_looking_count}")

    # Verification option
    verify = input("\nWould you like to verify annotations? (y/n): ")
    if verify.lower() == "y":
        incorrect_labels = dataset.verify_annotations()
        if incorrect_labels:
            print("\nIncorrect labels found:")
            for label in incorrect_labels:
                print(f"Dataset: {label['dataset']}")
                print(f"Path: {label['path']}")
                print(f"Current label: {label['current_label']}\n")

    # Test batch loading
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"Images: {batch['image'].shape}")
    print(f"Labels: {batch['is_looking'].shape}")
    print(f"Bboxes: {batch['bbox'].shape}")
