import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
#Usar para BCELoss y Sigmoid:
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")  # Grayscale mask
    return np.array(mask) // 255  # Convert 255 to 1

def predict_image(model, image_np):
    model.eval()
    h, w, _ = image_np.shape
    # Mantener normalización de entrada
    inputs_np = image_np.reshape(-1, 3) / 255.0
    inputs = torch.tensor(inputs_np, dtype=torch.float32)

    with torch.no_grad():
        # La salida del modelo (con Sigmoid) ya es la probabilidad P(skin)
        # Salida es [num_pixels, 1], usar squeeze() para [num_pixels]
        outputs = model(inputs).squeeze().numpy()

    # Outputs ya son las probabilidades de piel
    pred_probs_img = outputs.reshape(h, w)
    return pred_probs_img


"""
#Usar para CrossEntropyLoss y Softmax:
def predict_image(model, image_np):

    model.eval()
    h, w, _ = image_np.shape
    inputs_np = image_np.reshape(-1, 3) / 255.0  # Normalize to [0, 1] if needed
    inputs = torch.tensor(inputs_np, dtype=torch.float32)

    with torch.no_grad():
        # outputs are raw scores (logits), shape [num_pixels, 2]
        outputs = model(inputs)
        # Apply softmax to get probabilities, shape [num_pixels, 2]
        probs = torch.softmax(outputs, dim=1)
        # Extract probability for the 'skin' class (assuming index 1)
        # This is needed for ROC analysis based on skin probability
        skin_probs = probs[:, 1].numpy()

    # Reshape the skin probabilities back to image dimensions
    pred_probs_img = skin_probs.reshape(h, w)
    return pred_probs_img # Return the probability map for the skin class"""

def show_results(image_np, true_mask, pred_mask, threshold=0.5):
    pred_binary = (pred_mask >= threshold).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred_binary, cmap="gray")
    axs[2].set_title(f"Predicted Mask (Threshold={threshold})")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()