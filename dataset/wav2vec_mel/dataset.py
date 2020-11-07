from ..base import DatasetBase

import random


class Wav2VecMelDataset(DatasetBase):

    def __init__(self, data, indices=None):
        super().__init__(data)
        self.indices = indices

    def __getitem__(self, idx):
        i, z, src_mel = self.data[idx]
        if self.indices is None:
            return z, src_mel, src_mel
        _, _, tgt_mel = self.data[random.choice(self.indices[i])]
        return z, src_mel, tgt_mel
