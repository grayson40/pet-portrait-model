import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dataset_loader import PetPortraitDataset
from model import PetPortraitModel

class PetPortraitTrainer:
    def __init__(self, data_dir, batch_size=32, learning_rate=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = PetPortraitModel(pretrained=True).to(self.device)
        
        # Load dataset
        dataset = PetPortraitDataset(data_dir)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Define loss functions
        self.looking_criterion = nn.BCELoss()  # Binary Cross Entropy for looking/not looking
        self.quality_criterion = nn.MSELoss()  # Mean Squared Error for pose quality
        self.keypoint_criterion = nn.MSELoss()  # MSE for keypoint positions
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        
        # Save best model path
        self.best_model_path = Path('models/best_model.pth')
        self.best_model_path.parent.mkdir(exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_looking_acc = 0
        total_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Get batch data
            images = batch['image'].to(self.device)
            looking_targets = batch['is_looking'].float().to(self.device)  # Shape: (batch_size,)
            quality_targets = batch['pose_quality'].to(self.device)
            keypoint_targets = batch['keypoints'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate losses
            looking_loss = self.looking_criterion(
                outputs['looking'],  # Shape: (batch_size,)
                looking_targets     # Shape: (batch_size,)
            )
            quality_loss = self.quality_criterion(
                outputs['pose_quality'],
                quality_targets
            )
            keypoint_loss = self.keypoint_criterion(
                outputs['keypoints'],
                keypoint_targets[:, :, :2]
            )
            
            # Combine losses
            loss = looking_loss + quality_loss + 0.1 * keypoint_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy for looking prediction
            looking_preds = (outputs['looking'] > 0.5).float()
            accuracy = (looking_preds == looking_targets).float().mean()
            
            # Update metrics
            total_loss += loss.item()
            total_looking_acc += accuracy.item()
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'looking_acc': accuracy.item()
            })
        
        return total_loss / total_batches, total_looking_acc / total_batches

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_looking_acc = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                looking_targets = batch['is_looking'].float().to(self.device)  # Shape: (batch_size,)
                quality_targets = batch['pose_quality'].to(self.device)
                keypoint_targets = batch['keypoints'].to(self.device)
                
                outputs = self.model(images)
                
                looking_loss = self.looking_criterion(
                    outputs['looking'],  # Shape: (batch_size,)
                    looking_targets     # Shape: (batch_size,)
                )
                quality_loss = self.quality_criterion(
                    outputs['pose_quality'],
                    quality_targets
                )
                keypoint_loss = self.keypoint_criterion(
                    outputs['keypoints'],
                    keypoint_targets[:, :, :2]
                )
                
                loss = looking_loss + quality_loss + 0.1 * keypoint_loss
                
                looking_preds = (outputs['looking'] > 0.5).float()
                accuracy = (looking_preds == looking_targets).float().mean()
                
                total_loss += loss.item()
                total_looking_acc += accuracy.item()
                total_batches += 1
        
        return total_loss / total_batches, total_looking_acc / total_batches

    def train(self, num_epochs=50):
        """Train the model"""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.best_model_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Initialize trainer
    trainer = PetPortraitTrainer("data/raw", batch_size=32, learning_rate=1e-4)
    
    # Train model
    trainer.train(num_epochs=50)