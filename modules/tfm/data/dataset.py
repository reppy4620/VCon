import torch
from torch.utils.data import Dataset


# Simple dataset
class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.wav_to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan').fft.cpu()
        for p in self.wav_to_mel.parameters():
            p.requires_grad = False
        self.wav_to_mel.eval()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav = self.data[idx]
        mel = self.wav_to_mel(torch.tensor(wav, dtype=torch.float)[None, None, :]).squeeze(0)
        return wav, mel
