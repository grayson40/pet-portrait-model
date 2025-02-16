import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from balanced_dataset_loader import BalancedPetDataset
from model import PetPortraitModel


class PetPortraitTrainer:
    def __init__(self, data_dir, batch_size=64, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = PetPortraitModel(pretrained=True).to(self.device)

        # Load datasets
        train_dataset = BalancedPetDataset(data_dir, phase="train")
        val_dataset = BalancedPetDataset(data_dir, phase="val")

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Create data loaders with larger batch size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Single binary classification loss
        self.criterion = nn.BCELoss()

        # Optimizer with higher learning rate and gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        # More aggressive learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Watch accuracy instead of loss
            patience=2,
            factor=0.2,
            verbose=True,
        )

        # Save best model path
        self.best_model_path = Path("models/best_model.pth")
        self.best_model_path.parent.mkdir(exist_ok=True)
        self.best_accuracy = 0.0

    def train_epoch(self):
        """Simplified training for one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Get batch data
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["is_looking"].float().to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs["looking"], targets)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate accuracy
            preds = (outputs["looking"] > 0.5).float()
            accuracy = (preds == targets).float().mean()

            # Update metrics
            total_loss += loss.item()
            total_acc += accuracy.item()
            total_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}"}
            )

        return total_loss / total_batches, total_acc / total_batches

    def validate(self):
        """Simplified validation"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["is_looking"].float().to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs["looking"], targets)

                preds = (outputs["looking"] > 0.5).float()
                accuracy = (preds == targets).float().mean()

                total_loss += loss.item()
                total_acc += accuracy.item()
                total_batches += 1

        return total_loss / total_batches, total_acc / total_batches

    def train(self, num_epochs=10):
        """Faster training with early stopping"""
        print("\nStarting training...")
        print(f"Training on device: {self.device}")

        patience = 0
        max_patience = 4

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Update learning rate based on accuracy
            self.scheduler.step(val_acc)

            # Save best model based on accuracy
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "accuracy": val_acc,
                    },
                    self.best_model_path,
                )
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
                patience = 0
            else:
                patience += 1

            # Early stopping
            if patience >= max_patience:
                print(f"\nStopping early - no improvement for {max_patience} epochs")
                break


if __name__ == "__main__":
    # Initialize trainer with faster training settings
    trainer = PetPortraitTrainer(
        "data/raw",
        batch_size=64,  # Larger batch size
        learning_rate=1e-3,  # Higher learning rate
    )

    # Train for fewer epochs with early stopping
    trainer.train(num_epochs=10)
