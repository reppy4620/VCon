from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import Wav2VecMelDataset
from ..base import DataModuleBase
from ..common import load_data_with_indices


class Wav2VecMelDataModule(DataModuleBase):

    def setup(self, stage=None):
        train, valid, train_i, valid_i = load_data_with_indices(Path(self.params.data_dir), ratio=self.params.train_ratio)
        self.train_x, self.valid_x = Wav2VecMelDataset(train, train_i), Wav2VecMelDataset(valid)

    def _collate_fn(self, batch):
        src, src_mel, tgt_mel = zip(*batch)
        src_lens = [x.size(0) for x in src]
        tgt_lens = [x.size(0) for x in tgt_mel]
        overlap_len = [min(src_len, tgt_len) for src_len, tgt_len in zip(src_lens, tgt_lens)]

        src = pad_sequence(src, batch_first=True)
        src_mask = torch.stack([torch.arange(src.size(1)) < l for l in src_lens])

        tgt_mel = pad_sequence(tgt_mel, batch_first=True, padding_value=-5)
        tgt_mel = tgt_mel.transpose(1, 2)
        tgt_mask = torch.stack([torch.arange(tgt_mel.size(1)) < l for l in tgt_lens])
        return src, tgt_mel, src_mel, src_mask, tgt_mask, overlap_len
