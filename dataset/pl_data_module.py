from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils import load_data
from .collate_fn import collate_fn
from .dataset import VCDataset, VCDatasetFromPath


class VConDataModule(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        # following variable will be initialized in setup function.
        self.train_x = None
        self.valid_x = None

    def setup(self, stage=None):
        train, valid = load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
        self.train_x, self.valid_x = VCDataset(train), VCDataset(valid)

    def train_dataloader(self):
        return DataLoader(
            self.train_x,
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_x,
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4
        )
