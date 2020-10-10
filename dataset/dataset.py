import pathlib

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import get_wav


# Simple dataset
class VCDataset(Dataset):
    def __init__(self, data):
        # data = [raw1, ...]
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


# This dataset for the person that doesn't want to run preprocess.py.
class VCDatasetFromPath(Dataset):
    def __init__(self, fns):
        self.fns = fns
        self.wav_to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        wav = get_wav(self.fns[idx])
        mel = self.wav_to_mel(torch.tensor(wav, dtype=torch.float)[None, None, :]).squeeze(0)
        return wav, mel
