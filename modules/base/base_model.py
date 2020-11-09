import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import denormalize


class BaseModel(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        self.is_normalize = params.is_normalize

    # for training
    def forward(self, *args):
        raise NotImplementedError

    # for inference
    def inference(self, *args):
        raise NotImplementedError

    @staticmethod
    def freeze(model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    @staticmethod
    def unsqueeze_for_input(x):
        # if x doesn't batch dim, unsqueeze spec
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(x.size()) != 3:
            raise ValueError("len(x.size()) must be 2 or 3")
        return x

    def _mel_to_wav(self, mel):
        if self.is_normalize:
            mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).squeeze(0).detach().cpu().numpy()
        return wav

    def inverse(self, mel):
        if len(mel.size()) == 2:
            mel = mel.unsqueeze(0)
        if self.is_normalize:
            mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).detach().cpu()
        # wav: torch.Tensor (1, L)
        return wav

    @staticmethod
    def _adjust_length(mel, freq):
        t_dim = mel.size(-1)
        mod_val = t_dim % freq
        if mod_val != 0:
            if mod_val < t_dim / 2:
                mel = mel[:, :t_dim - mod_val]
            else:
                pad_length = t_dim + freq - mod_val
                mel = F.pad(mel[None, :, :], [0, pad_length], value=-5).squeeze(0)
        return mel
