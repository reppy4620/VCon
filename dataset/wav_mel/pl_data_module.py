from pathlib import Path

import torch

from .dataset import WavMelDataset, WavMelFromFileDataset
from ..base import DataModuleBase
from ..common import load_fns, load_data


class WavMelDataModule(DataModuleBase):

    def setup(self, stage=None):
        if self.params.from_fn:
            train, valid = load_fns(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = WavMelFromFileDataset(train), WavMelFromFileDataset(valid)
        else:
            train, valid = load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = WavMelDataset(train), WavMelDataset(valid)

    def _collate_fn(self, batch):
        wav, mel = tuple(zip(*batch))
        mel = torch.stack([self._preprocess(x) for x in mel])
        # wav: list(numpy 1d-array)
        # mel: torch.FloatTensor - (Batch, Mel-bin, Time)
        return wav, mel
