import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler



data = np.load('data/skin_nskin.npy')

# Split the data into training and testing sets
df_train, df_test = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True, random_state= 42)
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

class SkinDataset(Dataset):
    def __init__(self, data):
        self.data = data
        super(SkinDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, 0:3], dtype=torch.float32)
        y = torch.tensor(self.data[idx, 3], dtype=torch.float32)
        return x, y
    
# Create dataloaders
# Use WeightedRandomSampler to balance the dataset
# testing weights

labels = df_train[:, 3]  # The last column is the label
class_counts = np.bincount(labels.astype(int))
print("Class counts:", class_counts)
class_weights = 1. / class_counts
sample_weights = class_weights[labels.astype(int)]


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # usually same as dataset size
    replacement=True  # allows resampling
)

train_dataset = SkinDataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

val_dataset = SkinDataset(df_test)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class MLP(nn.Module):

    def __init__(self, input_size=3, hidden1_size= 8, hidden2_size=16, output_size= 1):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        self.hidden_layer_1 = nn.Linear(input_size, hidden1_size) # input -> hidden1
        self.hidden_layer_2 = nn.Linear(hidden1_size, hidden2_size) # hidden1 -> hidden2
        self.output_layer = nn.Linear(hidden2_size, output_size) # hidden2 -> output

         # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #x = self.flatten(x)
        x = self.relu(self.hidden_layer_1(x))
        x = self.relu(self.hidden_layer_2(x))
        x = self.sigmoid(self.output_layer(x))
        return x
    
model = MLP()
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr= 0.01)

num_epochs = 10  # you can start with a small number

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # remove extra dimension

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        preds = (outputs >= 0.5).float()  # binary classification threshold
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = train_correct / train_total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss/train_total:.4f} "
          f"Train Acc: {train_accuracy:.4f} "
          f"Val Loss: {val_loss/val_total:.4f} "
          f"Val Acc: {val_accuracy:.4f}")