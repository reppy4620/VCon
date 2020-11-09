import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from abc import abstractmethod
from torch.utils.data import DataLoader
from utils import normalize


class DataModuleBase(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        # following variable will be initialized in setup function.
        self.train_x = None
        self.valid_x = None

        self.is_normalize = params.is_normalize

    def train_dataloader(self):
        return DataLoader(
            self.train_x,
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_x,
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=4
        )

    def _preprocess(self, x):
        if x.size(-1) < self.params.seq_len:
            x = F.pad(x[None, :, :], [0, self.params.seq_len - x.size(-1)], mode='replicate').squeeze(0)
        max_offset = x.size(-1) - self.params.seq_len
        # generate int value from range of 0 <= sig_offset <= max_offset
        # maybe, np.random.randint is faster than pure python's random.randint
        sig_offset = np.random.randint(0, max_offset + 1)
        x = x[:, sig_offset:sig_offset + self.params.seq_len]
        if self.is_normalize:
            x = normalize(x)
        return x.float()

    @abstractmethod
    def _collate_fn(self, batch):
        raise NotImplementedError
