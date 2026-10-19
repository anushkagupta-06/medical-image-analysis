
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "vgg16_best.pth"
)


# --------------------------------------------------
# LOAD TRAINED VGG16
# --------------------------------------------------

def load_model():
    # Recreate the architecture used during training.
    model = models.vgg16(weights=None)

    # Adapt the first convolution for grayscale images.
    old_conv = model.features[0]

    model.features[0] = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding
    )

    # Two output classes: NORMAL and PNEUMONIA.
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        2
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"VGG16 checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# Load once when the Flask application starts.
model = load_model()


# --------------------------------------------------
# PREDICTION + GRAD-CAM
# --------------------------------------------------

def predict_xray(image_path, heatmap_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read the uploaded X-ray.")

    # Match the input dimensions and scaling used in Colab.
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(DEVICE)
    tensor.requires_grad_(True)

    # Capture the final convolutional feature maps.
    activations = {}

    def capture_features(module, inputs, output):
        activations["features"] = output

    target_layer = model.features[28]
    hook = target_layer.register_forward_hook(capture_features)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients are needed for Grad-CAM.
        with torch.enable_grad():
            outputs = model(tensor)

            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = int(outputs.argmax(dim=1).item())
            confidence = float(probabilities[predicted_class].item())

            feature_maps = activations["features"]

            gradients = torch.autograd.grad(
                outputs=outputs[0, predicted_class],
                inputs=feature_maps
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = torch.relu(
                (weights * feature_maps).sum(dim=1)
            ).squeeze(0)

            cam = cam.detach().cpu().numpy()

        # Normalize the heatmap.
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        # Create a coloured heatmap and overlay.
        heatmap = cv2.applyColorMap(
            np.uint8(cam * 255),
            cv2.COLORMAP_JET
        )

        original = np.uint8(image * 255)
        original_bgr = cv2.cvtColor(
            original, cv2.COLOR_GRAY2BGR
        )

        overlay = cv2.addWeighted(
            original_bgr, 0.6,
            heatmap, 0.4, 0
        )

        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)

        if not cv2.imwrite(heatmap_path, overlay):
            raise OSError("Could not save the Grad-CAM image.")

        return {
            "label": (
                "PNEUMONIA" if predicted_class == 1 else "NORMAL"
            ),
            "confidence": round(confidence * 100, 2),
            "heatmap_path": heatmap_path
        }

    finally:
        hook.remove()
