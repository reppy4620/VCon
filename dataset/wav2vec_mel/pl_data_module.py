from pathlib import Path

import torch

from .dataset import Wav2VecMelDataset
from ..base import DataModuleBase
from ..common import load_data_with_indices


class Wav2VecMelDataModule(DataModuleBase):

    def setup(self, stage=None):
        train, valid, train_i, valid_i = load_data_with_indices(Path(self.params.data_dir), ratio=self.params.train_ratio)
        self.train_x, self.valid_x = Wav2VecMelDataset(train, train_i), Wav2VecMelDataset(valid)

    def _collate_fn(self, batch):
        z, mel = zip(*batch)
        mel = torch.stack([self._preprocess(x) for x in mel])
        # z: torch.FloatTensor - (B, T, C)
        # tgt: torch.FloatTensor - (B, Mel-bin, T)
        return z, mel
