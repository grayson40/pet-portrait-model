import torch
import torch.nn as nn
import torch.quantization
import onnx
import tensorflow as tf
from model import PetPortraitModel


def optimize_for_mobile():
    # Load your trained model
    model = PetPortraitModel(pretrained=False)
    checkpoint = torch.load("models/best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 1. Export to ONNX (intermediate format)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "models/pet_portrait.onnx",
        input_names=["input"],
        output_names=["looking", "pose_quality", "keypoints", "bbox"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "looking": {0: "batch_size"},
            "pose_quality": {0: "batch_size"},
            "keypoints": {0: "batch_size"},
            "bbox": {0: "batch_size"},
        },
    )

    # 2. Convert ONNX to TFLite
    # Load ONNX model
    onnx_model = onnx.load("models/pet_portrait.onnx")

    # Convert to TensorFlow
    import tf2onnx

    tf_rep = tf2onnx.convert.from_onnx(onnx_model)
    tf_model = tf_rep[0]

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Enable quantization
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # Convert model
    tflite_model = converter.convert()

    # Save model
    with open("models/pet_portrait.tflite", "wb") as f:
        f.write(tflite_model)

    print(
        "Model size before optimization:",
        os.path.getsize("models/pet_portrait.onnx") / (1024 * 1024),
        "MB",
    )
    print(
        "Model size after optimization:",
        os.path.getsize("models/pet_portrait.tflite") / (1024 * 1024),
        "MB",
    )


if __name__ == "__main__":
    optimize_for_mobile()
