import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import PetPortraitModel
from balanced_dataset_loader import BalancedPetDataset


class ModelEvaluator:
    def __init__(self, model_path, data_dir, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = PetPortraitModel(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load validation dataset
        self.dataset = BalancedPetDataset(data_dir, phase="val")
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    def evaluate_accuracy(self):
        """Evaluate model accuracy on validation set"""
        total_looking = 0
        total_samples = 0
        keypoint_errors = []
        bbox_errors = []
        pose_quality_errors = []

        print("\nEvaluating model accuracy...")
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                images = batch["image"].to(self.device)
                looking_targets = batch["is_looking"].float().to(self.device)
                dataset_source = batch.get("dataset", [""] * images.shape[0])

                # Get predictions
                outputs = self.model(images, dataset_source)
                looking_preds = (outputs["looking"] > 0.5).float()

                # Update looking accuracy
                total_looking += (looking_preds == looking_targets).sum().item()
                total_samples += images.shape[0]

                # For animal pose samples, evaluate additional metrics
                animal_pose_mask = torch.tensor(
                    [ds == "animal_pose" for ds in dataset_source], device=self.device
                )
                if animal_pose_mask.any():
                    # Get animal pose targets
                    keypoint_targets = batch["keypoints"][animal_pose_mask].to(self.device)
                    bbox_targets = batch["bbox"][animal_pose_mask].to(self.device)
                    pose_quality_targets = batch["pose_quality"][animal_pose_mask].to(self.device)

                    # Calculate errors for animal pose samples
                    keypoint_error = torch.norm(
                        outputs["keypoints"][animal_pose_mask, :, :2] - keypoint_targets[:, :, :2],
                        dim=2
                    ).mean().item()
                    keypoint_errors.append(keypoint_error)

                    bbox_error = torch.norm(
                        outputs["bbox"][animal_pose_mask] - bbox_targets
                    ).mean().item()
                    bbox_errors.append(bbox_error)

                    pose_quality_error = torch.abs(
                        outputs["pose_quality"][animal_pose_mask] - pose_quality_targets
                    ).mean().item()
                    pose_quality_errors.append(pose_quality_error)

        # Calculate final metrics
        looking_accuracy = total_looking / total_samples
        avg_keypoint_error = np.mean(keypoint_errors) if keypoint_errors else 0
        avg_bbox_error = np.mean(bbox_errors) if bbox_errors else 0
        avg_pose_quality_error = np.mean(pose_quality_errors) if pose_quality_errors else 0

        return {
            "looking_accuracy": looking_accuracy,
            "keypoint_error": avg_keypoint_error,
            "bbox_error": avg_bbox_error,
            "pose_quality_error": avg_pose_quality_error
        }

    def evaluate_inference_speed(self, num_iterations=100):
        """Evaluate model inference speed"""
        print("\nEvaluating inference speed...")
        batch = next(iter(self.dataloader))
        images = batch["image"].to(self.device)

        # Warm up
        for _ in range(10):
            _ = self.model(images)

        # Time inference
        if torch.cuda.is_available():
            # GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_iterations):
                _ = self.model(images)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / num_iterations
        else:
            # CPU timing
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = self.model(images)
            elapsed_time = (time.perf_counter() - start_time) * 1000 / num_iterations  # Convert to ms

        return {
            "avg_inference_time": elapsed_time,
            "fps": 1000 / elapsed_time
        }


if __name__ == "__main__":
    # Initialize evaluator
    model_path = "models/best_model.pth"
    data_dir = "data/raw"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        exit(1)

    evaluator = ModelEvaluator(model_path, data_dir)

    # Evaluate accuracy
    accuracy_metrics = evaluator.evaluate_accuracy()
    print("\nAccuracy Metrics:")
    print(f"Looking Accuracy: {accuracy_metrics['looking_accuracy']:.4f}")
    print(f"Keypoint Error: {accuracy_metrics['keypoint_error']:.4f}")
    print(f"Bbox Error: {accuracy_metrics['bbox_error']:.4f}")
    print(f"Pose Quality Error: {accuracy_metrics['pose_quality_error']:.4f}")

    # Evaluate speed
    speed_metrics = evaluator.evaluate_inference_speed()
    print("\nSpeed Metrics:")
    print(f"Average Inference Time: {speed_metrics['avg_inference_time']:.2f} ms")
    print(f"Frames Per Second: {speed_metrics['fps']:.2f}")

    # Save metrics to file
    Path("models").mkdir(exist_ok=True)
    with open("models/evaluation_results.txt", "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")

        f.write("Accuracy Metrics:\n")
        f.write(f"Looking Accuracy: {accuracy_metrics['looking_accuracy']:.4f}\n")
        f.write(f"Keypoint Error: {accuracy_metrics['keypoint_error']:.4f}\n")
        f.write(f"Bbox Error: {accuracy_metrics['bbox_error']:.4f}\n")
        f.write(f"Pose Quality Error: {accuracy_metrics['pose_quality_error']:.4f}\n")
        f.write("\n")

        f.write("Speed Metrics:\n")
        f.write(f"Average Inference Time: {speed_metrics['avg_inference_time']:.2f} ms\n")
        f.write(f"Frames Per Second: {speed_metrics['fps']:.2f}\n")
