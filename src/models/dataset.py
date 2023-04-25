import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.Tensor(self.data[idx]), torch.Tensor(self.labels[idx])
        else:
            return torch.Tensor(self.data[idx])