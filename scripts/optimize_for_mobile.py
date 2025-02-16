import torch
import tensorflow as tf
import os
from model import PetPortraitModel
import numpy as np
import json


def hardswish(x):
    """Hardswish activation function"""
    return x * tf.nn.relu6(x + 3) / 6


def optimize_for_mobile():
    print("Loading PyTorch model...")
    try:
        # Load your trained PyTorch model
        model = PetPortraitModel(pretrained=False)
        checkpoint = torch.load("models/best_model.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("Creating TensorFlow model...")
        # Create equivalent TF model using MobileNetV3-Small
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(192, 192, 3), include_top=False, weights=None
        )

        # Create custom top to match your PyTorch model
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Lambda(hardswish)(x)  # Custom hardswish activation
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        tf_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Save as Keras model first
        os.makedirs("models", exist_ok=True)
        tf_model.save("models/tf_model.keras")

        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

        def representative_dataset():
            for _ in range(100):
                # Ensure data is float32
                data = np.random.rand(1, 192, 192, 3).astype(np.float32)
                # Normalize using float32 values
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                data = ((data - mean) / std).astype(np.float32)
                yield [data]

        # Configure optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.target_spec.supported_types = [tf.int8]

        print("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # Save TFLite model
        output_path = "models/pet_portrait.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # Generate metadata for mobile app
        metadata = {
            "name": "Pet Portrait Looking Detection",
            "version": "1.0",
            "input_shape": [1, 192, 192, 3],
            "input_type": "uint8",
            "output_shape": [1],
            "output_type": "uint8",
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225],
            "threshold": 0.7,  # From your predict_optimal_moment method
        }

        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\nOptimization complete!")
        print(f"Model size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print("\nFiles generated:")
        print("1. models/pet_portrait.tflite - Optimized model for mobile")
        print("2. models/model_metadata.json - Model metadata for mobile app")

        # Verify model
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\nModel Details:")
        print(f"Input Shape: {input_details[0]['shape']}")
        print(f"Input Type: {input_details[0]['dtype']}")
        print(f"Output Shape: {output_details[0]['shape']}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    optimize_for_mobile()
