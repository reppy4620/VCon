import numpy as np
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

    def _to_mel(self, wav):
        wav = np.pad(wav, [768, 768], 'reflect')
        return self.wav_to_mel(torch.tensor(wav, dtype=torch.float)[None, :]).squeeze(0)

    def __getitem__(self, idx):
        wav = self.data[idx]
        mel = self._to_mel(wav)
        return mel, mel.clone()
