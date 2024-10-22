import librosa
import torch
import pyworld as pw

from scipy.io.wavfile import write
from transforms import Wav2Mel


# load raw wav
def get_wav(fn):
    wav, sr = librosa.load(fn)
    # wav = trim_long_silences(wav)
    return wav


# load raw wav and mel-spectrogram from fn by using official melgan's converter
def get_wav_mel(fn, to_mel=None):
    if to_mel is None:
        to_mel = Wav2Mel()
    wav = get_wav(fn)
    mel = to_mel(torch.tensor(wav, dtype=torch.float)).squeeze(0)
    return wav, mel.cpu()


def get_world_features(fn, sr=22050):
    wav = get_wav(fn)
    f0, sp, ap = pw.wav2world(wav, sr)
    return f0, sp.T, ap.T


def get_wav2vec_features(fn, wav2vec, to_mel=None):
    wav, mel = get_wav_mel(fn, to_mel)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav = torch.tensor(wav, dtype=torch.float, device=device)[None, :]
    with torch.no_grad():
        feat = wav2vec.extract_features(wav, None)[0]
        feat = feat.detach().cpu().squeeze(0)
    return feat, mel


def normalize(x):
    return (torch.clamp(x, min=-5, max=1) + 2.5) / 2.5


def denormalize(y):
    return 2.5 * y - 2.5


# save sample
def save_sample(file_path, audio, sr=22050, normalize=True):
    if normalize:
        audio = (audio * 32768).astype("int16")
    write(file_path, sr, audio)
