from torch.utils.data import Dataset


# Simple dataset
class VCDataset(Dataset):
    def __init__(self, data):
        # data = [(raw1, mel1), ...]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, mel = self.data[idx]
        return wav, mel
