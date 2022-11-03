import torch


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=1_000, add=0):
        self.n = n
        self.add = add
        self.db = torch.arange(n)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        return self.db[idx] + self.add
