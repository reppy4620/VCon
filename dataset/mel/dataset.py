from ..base import DatasetBase
from utils import get_wav

import random


class MelDataset(DatasetBase):

    def __init__(self, data, indices=None):
        super().__init__(data)
        self.indices = indices

    def __getitem__(self, idx):
        i, src_wav = self.data[idx]
        src_mel = self._to_mel(src_wav)
        if self.indices is None:
            return src_mel, src_mel

        _, tgt_wav = self.data[random.choice(self.indices[i])]
        tgt_mel = self._to_mel(tgt_wav)
        return src_mel, tgt_mel


class MelFromFileDataset(DatasetBase):
    def __init__(self, data, indices=None):
        super().__init__(data)
        self.indices = indices

    def __getitem__(self, idx):
        i, src_fn = self.data[idx]
        src_wav = get_wav(src_fn)
        src_mel = self._to_mel(src_wav)
        if self.indices is None:
            return src_mel, src_mel

        _, tgt_fn = self.data[random.choice(self.indices[i])]
        tgt_wav = get_wav(tgt_fn)
        tgt_mel = self._to_mel(tgt_wav)
        return src_mel, tgt_mel
