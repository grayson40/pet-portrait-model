import torch
import torch.nn as nn
import torchvision.models as models


class PetPortraitModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)

        # Modify the classifier for our binary task
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[-1].in_features

        # Replace classifier with our own
        self.backbone.classifier = nn.Sequential(
            # First maintain the original structure
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            # Then add our final layer
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Forward pass through backbone and squeeze output
        return {"looking": self.backbone(x).squeeze(-1)}

    def predict_optimal_moment(self, x, threshold=0.7):
        """Real-time prediction for optimal photo moment"""
        with torch.no_grad():
            predictions = self(x)
            is_looking = predictions["looking"] > threshold
            return is_looking, predictions


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

    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create sample batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # Test forward pass
    outputs = model(x)
    print("\nOutput shapes:")
    for k, v in outputs.items():
        print(f"{k}: {tuple(v.shape)}")

    # Test prediction
    is_looking, predictions = model.predict_optimal_moment(x)
    print(f"\nPrediction shape: {is_looking.shape}")

    # Memory usage
    print("\nMemory Usage:")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
