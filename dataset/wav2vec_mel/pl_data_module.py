import torch

from pathlib import Path

from .dataset import Wav2VecMelDataset
from ..base import DataModuleBase
from ..common import load_data


class Wav2VecMelDataModule(DataModuleBase):

    def setup(self, stage=None):
        train, valid = load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
        self.train_x, self.valid_x = Wav2VecMelDataset(train), Wav2VecMelDataset(valid)

    def _collate_fn(self, batch):
        src, tgt = tuple(zip(*batch))
        src = torch.stack([self._preprocess(x) for x in src])
        tgt = torch.stack([self._preprocess(x) for x in tgt])
        # src: torch.FloatTensor - (Batch, Mel-bin, Time)
        # tgt: torch.FloatTensor - (Batch, Mel-bin, Time)
        return src, tgt
