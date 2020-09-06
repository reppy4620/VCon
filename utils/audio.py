import numpy as np
import torch

from resemblyzer import preprocess_wav
from librosa.feature import melspectrogram


# Use this params because resemblyzer use this params in making speaker embedding.
sampling_rate = 16000
mel_window_length = 25
mel_window_step = 10
mel_n_channels = 80


# wav to mel-spectrogram
# if pt == True, return torch.FloatTensor
# else return np.ndarray
def wav_to_mel(wav, pt=True):
    frames = melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return torch.tensor(frames.astype(np.float32).T, dtype=torch.float) if pt else frames.astype(np.float32).T


# get wav and mel-spectrogram
def get_wav_mel(fn, pt=True):
    wav = preprocess_wav(fn)
    mel = wav_to_mel(wav, pt=pt)
    if pt:
        wav = torch.tensor(wav, dtype=torch.float)
    return wav, mel
