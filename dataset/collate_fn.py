import torch
import torch.nn.functional as F


def collate_fn(batch, padding=False):
    wav, mel = tuple(zip(*batch))
    if padding:
        # padding
        max_length = max(map(lambda x: x.size(-1), mel))
        mel = torch.stack(tuple(map(lambda x: F.pad(x, (0, max_length - x.size(-1))), mel)))
    else:
        # drop
        min_length = min(map(lambda x: x.size(-1), mel))
        mel = torch.stack(tuple(map(lambda x: x[:, :min_length], mel)))
    return wav, mel
