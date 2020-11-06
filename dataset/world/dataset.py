import torch

from ..base import DatasetBase
from utils import get_world_feature


# Simple dataset
class WorldDataset(DatasetBase):
    def __getitem__(self, idx):
        f0, sp, _ = self.data[idx]
        return f0, sp


class WorldFromFileDataset(DatasetBase):
    def __getitem__(self, idx):
        fn = self.data[idx]
        f0, sp, _ = get_world_feature(fn)
        f0, sp = torch.tensor(f0, dtype=torch.float), torch.tensor(sp, dtype=torch.float)
        return f0, sp
