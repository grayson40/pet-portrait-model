import torch
import torch.nn as nn
import torchvision.models as models


class PetPortraitModel(nn.Module):
    def __init__(self, pretrained=True):
        super(PetPortraitModel, self).__init__()

        # Load pretrained ResNet18 but freeze some layers to reduce overfitting
        self.backbone = models.resnet18(pretrained=pretrained)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-4]:  # Keep last block trainable
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Reduced capacity looking head with increased regularization
        self.looking_head = nn.Sequential(
            nn.Linear(num_features, 512),  # Reduced from 1024
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, 256),  # Added intermediate layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Simplified pose quality head
        self.pose_quality_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Simplified keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Linear(num_features, 512),  # Reduced from 1024
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 20 * 3),
        )

        # Simplified bbox head
        self.bbox_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
        )

    def forward(self, x, dataset_source=None):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        # Primary task: looking classification
        looking = self.looking_head(features)

        # Initialize outputs
        outputs = {"looking": looking.squeeze(-1)}

        # Add Animal Pose specific outputs if needed
        if dataset_source == "animal_pose":
            outputs.update(
                {
                    "pose_quality": self.pose_quality_head(features).squeeze(-1),
                    "keypoints": self.keypoint_head(features).view(-1, 20, 3),
                    "bbox": self.bbox_head(features),
                }
            )
        else:
            batch_size = x.size(0)
            outputs.update(
                {
                    "pose_quality": torch.zeros(batch_size, device=x.device),
                    "keypoints": torch.zeros(batch_size, 20, 3, device=x.device),
                    "bbox": torch.zeros(batch_size, 4, device=x.device),
                }
            )

        return outputs

    def predict_optimal_moment(self, x, threshold=0.7):
        with torch.no_grad():
            predictions = self(x)
            is_looking = predictions["looking"] > threshold
            return is_looking, predictions

    def get_loss_weights(self, batch_size, dataset_source):
        # Reduced weights for auxiliary tasks
        if dataset_source == "animal_pose":
            return {
                "looking": 1.0,
                "pose_quality": 0.3,  # Reduced from 0.5
                "keypoints": 0.2,  # Reduced from 0.3
                "bbox": 0.2,  # Reduced from 0.3
            }
        else:
            return {"looking": 1.0, "pose_quality": 0.0, "keypoints": 0.0, "bbox": 0.0}


if __name__ == "__main__":
    # Initialize model
    model = PetPortraitModel(pretrained=True)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create sample batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # Test regular forward pass
    print("\nRegular forward pass:")
    outputs = model(x)
    for k, v in outputs.items():
        print(f"{k}: {tuple(v.shape)}")

    # Test animal pose forward pass
    print("\nAnimal pose forward pass:")
    outputs = model(x, dataset_source="animal_pose")
    for k, v in outputs.items():
        print(f"{k}: {tuple(v.shape)}")

    # Test prediction
    print("\nTesting prediction:")
    is_looking, predictions = model.predict_optimal_moment(x)
    print(f"Is looking shape: {is_looking.shape}")

    # Test loss weights
    print("\nLoss weights:")
    print("Regular dataset:", model.get_loss_weights(batch_size, None))
    print("Animal pose dataset:", model.get_loss_weights(batch_size, "animal_pose"))

    # Memory usage
    print("\nMemory Usage:")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
