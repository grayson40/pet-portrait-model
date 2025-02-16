import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from model import PetPortraitModel
from balanced_dataset_loader import BalancedPetDataset

# Try to import psutil, but don't require it
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not installed. Memory tracking will be limited.")


class ModelEvaluator:
    def __init__(
        self, model_path, data_dir, batch_size=1
    ):  # batch_size=1 for mobile simulation
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
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Reduced for mobile simulation
        )

    def get_memory_usage(self):
        """Get memory usage in MB"""
        if HAS_PSUTIL:
            return psutil.Process().memory_info().rss / 1024 / 1024
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0  # Fallback when no memory tracking is available

    def evaluate_accuracy(self):
        """Evaluate model accuracy and generate detailed metrics"""
        all_preds = []
        all_targets = []

        print("\nEvaluating model accuracy...")
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                images = batch["image"].to(self.device)
                targets = batch["is_looking"].to(self.device)

                # Forward pass
                outputs = self.model(images)
                preds = (outputs["looking"] > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Calculate additional metrics
        report = classification_report(all_targets, all_preds, output_dict=True)

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        }

    def evaluate_mobile_performance(self, num_iterations=100):
        """Evaluate model performance for mobile deployment"""
        print("\nEvaluating mobile performance metrics...")

        # Get sample image
        batch = next(iter(self.dataloader))
        image = batch["image"].to(self.device)

        # Memory usage before
        memory_before = self.get_memory_usage()

        # Warm up
        for _ in range(10):
            _ = self.model(image)

        # Measure inference times
        inference_times = []
        memory_usage = []

        for _ in range(num_iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = self.model(image)

            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            # Track memory
            memory = self.get_memory_usage()
            memory_usage.append(memory)

        # Calculate metrics
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        fps = 1000 / avg_inference

        memory_increase = np.mean(memory_usage) - memory_before

        # Model size
        model_size = (
            sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024
        )  # MB

        return {
            "avg_inference_ms": avg_inference,
            "std_inference_ms": std_inference,
            "fps": fps,
            "memory_mb": memory_increase,
            "model_size_mb": model_size,
            "p90_latency": np.percentile(inference_times, 90),
            "p95_latency": np.percentile(inference_times, 95),
        }


if __name__ == "__main__":
    model_path = "models/best_model.pth"
    data_dir = "data/raw"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        exit(1)

    evaluator = ModelEvaluator(model_path, data_dir)

    # Evaluate accuracy
    print("\nEvaluating model accuracy...")
    accuracy_metrics = evaluator.evaluate_accuracy()

    # Evaluate mobile performance
    print("\nEvaluating mobile performance...")
    mobile_metrics = evaluator.evaluate_mobile_performance()

    # Save comprehensive results
    results_path = Path("models/mobile_evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("Mobile Model Evaluation Results\n")
        f.write("=============================\n\n")

        f.write("Accuracy Metrics:\n")
        f.write(f"Accuracy: {accuracy_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {accuracy_metrics['precision']:.4f}\n")
        f.write(f"Recall: {accuracy_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {accuracy_metrics['f1']:.4f}\n\n")

        f.write("Mobile Performance Metrics:\n")
        f.write(f"Average Inference Time: {mobile_metrics['avg_inference_ms']:.2f}ms\n")
        f.write(f"Inference Std Dev: {mobile_metrics['std_inference_ms']:.2f}ms\n")
        f.write(f"Frames Per Second: {mobile_metrics['fps']:.2f}\n")
        f.write(f"90th Percentile Latency: {mobile_metrics['p90_latency']:.2f}ms\n")
        f.write(f"95th Percentile Latency: {mobile_metrics['p95_latency']:.2f}ms\n")
        f.write(f"Memory Usage: {mobile_metrics['memory_mb']:.2f}MB\n")
        f.write(f"Model Size: {mobile_metrics['model_size_mb']:.2f}MB\n")

    print(f"\nResults saved to {results_path}")
