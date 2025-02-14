import torch
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset_loader import PetPortraitDataset
from model import PetPortraitModel

class ModelEvaluator:
    def __init__(self, model_path, data_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = PetPortraitModel(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load test dataset
        self.dataset = PetPortraitDataset(data_dir)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,  # Increased batch size for faster evaluation
            shuffle=False
        )
        
    def evaluate_accuracy(self):
        """Evaluate model accuracy and generate detailed metrics"""
        print("\nEvaluating model accuracy...")
        
        all_looking_preds = []
        all_looking_targets = []
        all_quality_preds = []
        all_quality_targets = []
        all_keypoint_errors = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader):
                images = batch['image'].to(self.device)
                looking_targets = batch['is_looking'].to(self.device)
                quality_targets = batch['pose_quality'].to(self.device)
                keypoint_targets = batch['keypoints'].to(self.device)
                
                outputs = self.model(images)
                
                # Get predictions
                looking_preds = (outputs['looking'] > 0.5).float()
                
                # Convert to numpy arrays
                looking_preds_np = looking_preds.cpu().numpy()
                looking_targets_np = looking_targets.cpu().numpy()
                quality_preds_np = outputs['pose_quality'].cpu().numpy()
                quality_targets_np = quality_targets.cpu().numpy()
                
                # Store batch results
                all_looking_preds.append(looking_preds_np)
                all_looking_targets.append(looking_targets_np)
                all_quality_preds.append(quality_preds_np)
                all_quality_targets.append(quality_targets_np)
                
                # Calculate keypoint error if available
                if 'keypoints' in outputs:
                    keypoint_error = torch.mean(torch.norm(
                        outputs['keypoints'] - keypoint_targets[:, :, :2], 
                        dim=2
                    )).item()
                    all_keypoint_errors.append(keypoint_error)
        
        # Concatenate all batches
        all_looking_preds = np.concatenate(all_looking_preds)
        all_looking_targets = np.concatenate(all_looking_targets)
        all_quality_preds = np.concatenate(all_quality_preds)
        all_quality_targets = np.concatenate(all_quality_targets)
        
        # Calculate metrics
        looking_accuracy = np.mean(all_looking_preds == all_looking_targets)
        quality_mse = np.mean((all_quality_preds - all_quality_targets) ** 2)
        avg_keypoint_error = np.mean(all_keypoint_errors) if all_keypoint_errors else None
        
        # Generate classification report
        print("\nLooking at Camera Classification Report:")
        print(classification_report(all_looking_targets, all_looking_preds))
        
        # Create confusion matrix
        cm = confusion_matrix(all_looking_targets, all_looking_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        
        metrics = {
            'looking_accuracy': looking_accuracy,
            'quality_mse': quality_mse,
        }
        if avg_keypoint_error is not None:
            metrics['keypoint_error'] = avg_keypoint_error
            
        print("\nMetrics:")
        print(f"Looking Accuracy: {looking_accuracy:.4f}")
        print(f"Quality MSE: {quality_mse:.4f}")
        if avg_keypoint_error is not None:
            print(f"Average Keypoint Error: {avg_keypoint_error:.4f}")
        
        return metrics
    
    def test_inference_speed(self, num_runs=100):
        """Test model inference speed"""
        print("\nTesting inference speed...")
        
        # Use a fixed batch size for speed testing
        sample_batch = next(iter(self.data_loader))
        sample_images = sample_batch['image'].to(self.device)
        
        # Warm up
        for _ in range(10):
            _ = self.model(sample_images)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in tqdm(range(num_runs)):
                start_time = time.time()
                _ = self.model(sample_images)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                times.append(inference_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nInference Speed:")
        print(f"Average: {avg_time:.2f}ms")
        print(f"Std Dev: {std_time:.2f}ms")
        print(f"FPS: {1000/avg_time:.2f}")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': 1000/avg_time
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_path="models/best_model.pth",
        data_dir="data/raw"
    )
    
    # Run evaluations
    accuracy_metrics = evaluator.evaluate_accuracy()
    speed_metrics = evaluator.test_inference_speed()
    
    # Save metrics to file
    Path('models').mkdir(exist_ok=True)
    with open('models/evaluation_results.txt', 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")
        
        f.write("Accuracy Metrics:\n")
        f.write(f"Looking Accuracy: {accuracy_metrics['looking_accuracy']:.4f}\n")
        f.write(f"Quality MSE: {accuracy_metrics['quality_mse']:.4f}\n")
        if 'keypoint_error' in accuracy_metrics:
            f.write(f"Average Keypoint Error: {accuracy_metrics['keypoint_error']:.4f}\n")
        f.write("\n")
        
        f.write("Speed Metrics:\n")
        f.write(f"Average Inference Time: {speed_metrics['avg_time']:.2f}ms\n")
        f.write(f"Std Dev: {speed_metrics['std_time']:.2f}ms\n")
        f.write(f"FPS: {speed_metrics['fps']:.2f}\n")