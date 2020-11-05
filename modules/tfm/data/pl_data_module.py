from tqdm import tqdm
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .collate_fn import collate_fn
from .dataset import TransformerDataset


class TransformerDataModule(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        # following variable will be initialized in setup function.
        self.train_x = None
        self.valid_x = None

    # load data from preprocessed files
    def _load_data(self, data_dir: Path, ratio: float):
        fns = list(data_dir.glob('*.dat'))
        data = [torch.load(str(d)) for d in tqdm(fns, total=len(fns))]
        num_of_train = int(len(data) * ratio)
        train, valid = data[:num_of_train], data[num_of_train:]
        return sum(train, list()), sum(valid, list())

    def setup(self, stage=None):
        train, valid = self._load_data(Path(self.params.data_dir), ratio=self.params.train_ratio)
        self.train_x, self.valid_x = TransformerDataset(train), TransformerDataset(valid)

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
