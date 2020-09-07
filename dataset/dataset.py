from torch.utils.data import Dataset


class VCDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, _, mel = self.data[idx]
        return wav, mel
