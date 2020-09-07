import librosa
import torch


def load_wav(fn):
    data, sr = librosa.load(fn)
    data = 0.95 * librosa.util.normalize(data)
    return torch.from_numpy(data).float(), sr


def get_wav_mel(fn, to_mel):
    wav, sr = librosa.load(fn)
    wav = 0.95 * librosa.util.normalize(wav)
    mel = to_mel(torch.tensor(wav, dtype=torch.float).unsqueeze(0)).squeeze(0)
    return wav, mel.cpu()
