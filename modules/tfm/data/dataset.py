import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import Wav2Mel


# Simple dataset
class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.wav_to_mel = Wav2Mel()

    def __len__(self):
        return len(self.data)

    def _to_mel(self, wav):
        return self.wav_to_mel(torch.tensor(wav, dtype=torch.float)).squeeze(0)

    def __getitem__(self, idx):
        wav = self.data[idx]
        mel = self._to_mel(wav)
        return mel, mel.clone()
