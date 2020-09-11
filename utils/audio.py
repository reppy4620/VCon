import librosa
import torch

from resemblyzer import trim_long_silences
from scipy.io.wavfile import write


# load raw wav and mel-spectrogram from fn by using official melgan's converter
def get_wav_mel(fn, to_mel=None):
    if to_mel is None:
        to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, sr = librosa.load(fn)
    trimmed = trim_long_silences(wav)
    try:
        wav = 0.95 * librosa.util.normalize(trimmed)
    except:
        wav = 0.95 * librosa.util.normalize(wav)
    mel = to_mel(torch.tensor(wav, dtype=torch.float)[None, :]).squeeze(0)
    return wav, mel.cpu()


# save sample
def save_sample(file_path, audio, sr=22050):
    audio = (audio * 32768).astype("int16")
    write(file_path, sr, audio)
