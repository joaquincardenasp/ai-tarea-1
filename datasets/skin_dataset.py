import torch
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    def __init__(self, data):
        self.data = data
        super(SkinDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, 0:3], dtype=torch.float32) / 255.0  # Para Normalizar a [0, 1]
        #y = torch.tensor(self.data[idx, 3], dtype=torch.long)  # Usar para CrossEntropyLoss
        y = torch.tensor(self.data[idx, 3], dtype=torch.float32) #Usar para BCELoss y Sigmoid
        return x, y