import pathlib

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import get_wav_mel


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


class VCDatasetFromPath(Dataset):
    def __init__(self, data_dir):
        if isinstance(data_dir, pathlib.Path):
            data_dir = data_dir
        elif isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        else:
            raise ValueError('data_dir must be pathlib.Path or str')

        self.data = list()
        to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        fns = list(data_dir.glob('**/*.wav'))
        for fn in tqdm(fns, desc='Load Training Data', total=len(fns)):
            wav, mel = get_wav_mel(fn, to_mel=to_mel)
            self.data.append((wav, mel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, mel = self.data[idx]
        return wav, mel
