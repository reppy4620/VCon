import torch
import torch.nn.functional as F


# collate_fn which used in DataLoader
def collate_fn(batch, padding=False):
    wav, mel = tuple(zip(*batch))
    # adjust dim of time-dim to maximum length in batch(padding) or minimum length in batch(trim)
    if padding:
        # padding
        max_length = max(map(lambda x: x.size(-1), mel))
        mel = torch.stack(tuple(map(lambda x: F.pad(x, (0, max_length - x.size(-1))), mel)))
    else:
        # trim
        min_length = min(map(lambda x: x.size(-1), mel))
        mel = torch.stack(tuple(map(lambda x: x[:, :min_length], mel)))
    # wav: list(numpy 1d-array)
    # mel: torch.FloatTensor - (Batch, Mel-bin, Time)
    return wav, mel
