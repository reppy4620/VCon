import torch

from pathlib import Path

from .dataset import MelDataset, MelFromFileDataset
from ..base import DataModuleBase
from ..common import load_fns, load_data


class MelDataModule(DataModuleBase):

    def setup(self, stage=None):
        if self.params.from_fn:
            train, valid = load_fns(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = MelFromFileDataset(train), MelFromFileDataset(valid)
        else:
            train, valid = load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
            self.train_x, self.valid_x = MelDataset(train), MelDataset(valid)

    def _collate_fn(self, batch):
        src, tgt = tuple(zip(*batch))
        src = torch.stack([self._preprocess(x) for x in src])
        tgt = torch.stack([self._preprocess(x) for x in tgt])
        # src: torch.FloatTensor - (Batch, Mel-bin, Time)
        # tgt: torch.FloatTensor - (Batch, Mel-bin, Time)
        return src, tgt