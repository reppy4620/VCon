import numpy as np

import torch
import torch.nn.functional as F

from utils import normalize


# lazy parameter settings
_seq_len = 128  # 64 or 128


def _preprocess(x):
    if x.size(-1) < _seq_len:
        x = F.pad(x[None, :, :], [0, _seq_len - x.size(-1)], mode='replicate').squeeze(0)
    max_offset = x.size(-1) - _seq_len
    # generate int value from range of 0 <= sig_offset <= max_offset
    # maybe, np.random.randint is faster than pure python's random.randint
    sig_offset = np.random.randint(0, max_offset+1)
    x = x[:, sig_offset:sig_offset+_seq_len]
    x = normalize(x)
    return x.float()


# collate_fn which used in DataLoader
def collate_fn(batch):
    wav, mel = tuple(zip(*batch))
    mel = torch.stack([_preprocess(x) for x in mel])
    # wav: list(numpy 1d-array)
    # mel: torch.FloatTensor - (Batch, Mel-bin, Time)
    return wav, mel
