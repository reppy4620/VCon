from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from utils import load_data
from .collate_fn import collate_fn
from .dataset import VCDataset


class VConDataModule(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        # following variable will be initialized in setup function.
        self.train_x = None
        self.valid_x = None

    def setup(self, stage=None):
        data = load_data(Path(self.params.data_dir))
        dataset = VCDataset(data)
        # default train_ratio = 0.9
        train_size = int(self.params.train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        self.train_x, self.valid_x = random_split(
            dataset=dataset,
            lengths=[train_size, valid_size],
            generator=torch.Generator().manual_seed(self.params.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_x,
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_x,
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
