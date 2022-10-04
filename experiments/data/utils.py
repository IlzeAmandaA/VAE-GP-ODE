from torch.utils import data

# prepare dataset
class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # N,16,784
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx]

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
