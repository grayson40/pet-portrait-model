import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

from balanced_dataset_loader import BalancedPetDataset
from model import PetPortraitModel


class PetPortraitTrainer:
    def __init__(self, data_dir, batch_size=32, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = PetPortraitModel(pretrained=True).to(self.device)

        # Load datasets with phases
        train_dataset = BalancedPetDataset(data_dir, phase="train")
        val_dataset = BalancedPetDataset(data_dir, phase="val")

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Loss functions
        self.looking_criterion = nn.BCELoss()
        self.quality_criterion = nn.MSELoss()
        self.keypoint_criterion = nn.MSELoss()
        self.bbox_criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.1
        )

        # Save best model path
        self.best_model_path = Path("models/best_model.pth")
        self.best_model_path.parent.mkdir(exist_ok=True)
        self.best_val_loss = float("inf")

    def _compute_losses(self, outputs, batch, dataset_source):
        """Compute appropriate losses based on dataset source"""
        losses = {}

        # Looking loss (for all datasets)
        losses["looking"] = self.looking_criterion(
            outputs["looking"], batch["is_looking"].float().to(self.device)
        )

        # Animal Pose specific losses
        if dataset_source == "animal_pose":
            losses["quality"] = self.quality_criterion(
                outputs["pose_quality"], batch["pose_quality"].to(self.device)
            )
            losses["keypoints"] = self.keypoint_criterion(
                outputs["keypoints"], batch["keypoints"].to(self.device)
            )
            losses["bbox"] = self.bbox_criterion(
                outputs["bbox"], batch["bbox"].to(self.device)
            )

        # Combine losses with appropriate weights
        total_loss = losses["looking"]  # Primary task weight = 1.0
        if dataset_source == "animal_pose":
            total_loss += (
                0.5 * losses["quality"]
                + 0.3 * losses["keypoints"]
                + 0.3 * losses["bbox"]
            )

        return total_loss, losses

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_looking_acc = 0
        total_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Get batch data
            images = batch["image"].to(self.device)
            dataset_source = batch.get("dataset", "other")

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, dataset_source)

            # Compute losses
            loss, losses = self._compute_losses(outputs, batch, dataset_source)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy for looking prediction
            looking_preds = (outputs["looking"] > 0.5).float()
            accuracy = (
                (looking_preds == batch["is_looking"].float().to(self.device))
                .float()
                .mean()
            )

            # Update metrics
            total_loss += loss.item()
            total_looking_acc += accuracy.item()
            total_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "looking_acc": f"{accuracy.item():.4f}"}
            )

        return total_loss / total_batches, total_looking_acc / total_batches

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_looking_acc = 0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                dataset_source = batch.get("dataset", "other")

                outputs = self.model(images, dataset_source)
                loss, _ = self._compute_losses(outputs, batch, dataset_source)

                looking_preds = (outputs["looking"] > 0.5).float()
                accuracy = (
                    (looking_preds == batch["is_looking"].float().to(self.device))
                    .float()
                    .mean()
                )

                total_loss += loss.item()
                total_looking_acc += accuracy.item()
                total_batches += 1

        return total_loss / total_batches, total_looking_acc / total_batches

    def train(self, num_epochs=50):
        """Train the model"""
        print("\nStarting training...")
        print(f"Training on device: {self.device}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.best_model_path,
                )
                print(f"Saved best model with validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    # Initialize trainer
    trainer = PetPortraitTrainer("data/raw", batch_size=32, learning_rate=1e-4)

    # Train model
    trainer.train(num_epochs=50)
