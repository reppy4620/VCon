from pathlib import Path

import torch

from .dataset import MelDataset, MelFromFileDataset
from ..base import DataModuleBase
from ..common import load_data_with_indices, load_fns_with_indices


class MelDataModule(DataModuleBase):

    def setup(self, stage=None):
        if self.params.from_fn:
            train, valid, train_i, _ = load_fns_with_indices(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = MelFromFileDataset(train, train_i), MelFromFileDataset(valid)
        else:
            train, valid, train_i, _ = load_data_with_indices(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = MelDataset(train), MelDataset(valid)

    def _collate_fn(self, batch):
        src, tgt = tuple(zip(*batch))
        src = torch.stack([self._preprocess(x) for x in src])
        tgt = torch.stack([self._preprocess(x) for x in tgt])
        # src: torch.FloatTensor - (Batch, Mel-bin, Time)
        # tgt: torch.FloatTensor - (Batch, Mel-bin, Time)
        return src, tgt
