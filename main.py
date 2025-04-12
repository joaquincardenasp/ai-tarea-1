import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#Organice un poco en modulos tratando que quedara mas ordenado.
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from datasets.skin_dataset import SkinDataset
from utils.training_utils import train_one_epoch, validate
from models.mlp_1 import MLP
from utils.image_utils import load_image, load_mask, predict_image, show_results
from utils.roc_utils import compute_roc_curve, plot_roc_curve

#GET DATA FROM NUMPY FILE
data = np.load('data/skin_nskin.npy')

#SPLIT DATA
df_train, df_test = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True, random_state= 42)
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")


# DATALOADER
labels = df_train[:, 3]
class_counts = np.bincount(labels.astype(int))
print("Class counts:", class_counts)
if len(class_counts) > 0:
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
else:
    sampler = None

train_dataset = SkinDataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #Usar para caso sampler default
#train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler if sampler else None, shuffle=sampler is None) #Usar para caso WeightedRandomSampler
val_dataset = SkinDataset(df_test)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.BCELoss()
#loss_function = nn.CrossEntropyLoss()  # Cambiar a CrossEntropyLoss

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_function)
    val_loss, val_accuracy = validate(model, val_loader, loss_function)
    
    # Guarda métricas
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

torch.save(model.state_dict(), 'mlp.pth')
model = MLP()  # same architecture
model.load_state_dict(torch.load('mlp.pth'))
model.eval()

plt.figure(figsize=(12, 5))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Gráfico de exactitud
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

img_path = r"data\dataset_with_mask\dataset_with_mask\9_Press_Conference_Press_Conference_9_79.jpg"
mask_path = r"data\dataset_with_mask\dataset_with_mask\9_Press_Conference_Press_Conference_9_79_mask.png"

image = load_image(img_path)
print("Image shape (HWC):", np.array(image).shape)

mask = load_mask(mask_path)
print("Unique mask values:", np.unique(mask))


# Predict probabilities for each pixel
pred_probs = predict_image(model, image)
print("Predicted probs stats — min:", np.min(pred_probs), "max:", np.max(pred_probs), "mean:", np.mean(pred_probs))


# Show results with threshold=0.5
show_results(image, mask, pred_probs, threshold=0.5)

# ROC Curve
thresholds = np.arange(0.2, 1.0, 0.1)
fpr, tpr = compute_roc_curve(mask, pred_probs, thresholds)
plot_roc_curve(fpr, tpr, thresholds)

