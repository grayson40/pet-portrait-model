import torch
import torch.nn as nn
import torchvision.models as models

class PetPortraitModel(nn.Module):
    def __init__(self, pretrained=True):
        super(PetPortraitModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom heads for our tasks
        
        # 1. Looking at camera classification head
        self.looking_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # Single output for binary classification
            nn.Sigmoid()
        )
        
        # 2. Pose quality regression head
        self.pose_quality_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # 3. Keypoint regression head
        self.keypoint_head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 20 * 2)  # 20 keypoints, each with x,y coordinates
        )

    def forward(self, x):
        # Get features from backbone
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # Flatten the features
        
        # Get predictions from each head
        looking = self.looking_head(x)
        pose_quality = self.pose_quality_head(x)
        keypoints = self.keypoint_head(x).view(-1, 20, 2)  # Reshape to (batch_size, 20, 2)
        
        return {
            'looking': looking.squeeze(),  # Shape: (batch_size,)
            'pose_quality': pose_quality.squeeze(),  # Shape: (batch_size,)
            'keypoints': keypoints  # Shape: (batch_size, 20, 2)
        }

    def predict_optimal_moment(self, x, threshold=0.7):
        """Helper method to determine if this is a good moment to take a photo"""
        with torch.no_grad():
            predictions = self(x)
            is_looking = predictions['looking'] > threshold
            pose_quality = predictions['pose_quality']
            
            # Consider it a good moment if the pet is looking and pose quality is good
            optimal_moment = is_looking & (pose_quality > threshold)
            
            return optimal_moment, predictions

if __name__ == "__main__":
    # Test the model
    model = PetPortraitModel(pretrained=True)
    
    # Create a sample input (batch_size=2, channels=3, height=224, width=224)
    sample_input = torch.randn(2, 3, 224, 224)
    
    # Get predictions
    predictions = model(sample_input)
    
    # Print output shapes
    print("\nModel Output Shapes:")
    for key, value in predictions.items():
        print(f"{key}: {value.shape}")
    
    # Test optimal moment prediction
    optimal, preds = model.predict_optimal_moment(sample_input)
    print("\nOptimal Moment Shape:", optimal.shape)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Model size estimation (in MB)
    model_size = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    print(f"Estimated model size: {model_size:.2f} MB")