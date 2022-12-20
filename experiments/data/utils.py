from torch.utils import data
import torch

# prepare dataset
class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # N,16,784
        self.mean = 0.1307
        self.std = 0.3081
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        tensor = torch.tensor(self.Xtr[idx],dtype=torch.float32).view(16,1,28,28)
        t_n = (tensor - self.mean)/self.std
        return t_n 

class Dataset_labels(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.reshape(-1)
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x_idx = self.x[index]
        y_idx = self.y[index]

        return x_idx, y_idx
