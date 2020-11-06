from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from .dataset import WorldDataset, WorldFromFileDataset
from ..base import DataModuleBase
from ..common import load_fns, load_data


class WorldDataModule(DataModuleBase):

    def setup(self, stage=None):
        if self.params.from_fn:
            train, valid = load_fns(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = WorldFromFileDataset(train), WorldFromFileDataset(valid)
        else:
            train, valid = load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = WorldDataset(train), WorldDataset(valid)

    def _preprocess(self, f0, sp):
        if f0.size(-1) < self.params.seq_len:
            f0 = F.pad(f0[None, :, :], [0, self.params.seq_len - f0.size(-1)], mode='replicate').squeeze(0)
            sp = F.pad(sp[None, :, :], [0, self.params.seq_len - sp.size(-1)], mode='replicate').squeeze(0)
        max_offset = f0.size(-1) - self.params.seq_len
        # generate int value from range of 0 <= sig_offset <= max_offset
        # maybe, np.random.randint is faster than pure python's random.randint
        sig_offset = np.random.randint(0, max_offset + 1)
        f0 = f0[:, sig_offset:sig_offset + self.params.seq_len]
        sp = sp[:, sig_offset:sig_offset + self.params.seq_len]
        # x = normalize(x)
        return f0, sp

    def _collate_fn(self, batch):
        f0, sp = tuple(zip(*[self._preprocess(f0, sp) for f0, sp in zip(*batch)]))
        f0 = torch.stack(f0)
        sp = torch.stack(sp)
        # f0: torch.FloatTensor - (Batch, freq-bin, Time)
        # sp: torch.FloatTensor - (Batch, freq-bin, Time)
        return f0, sp
