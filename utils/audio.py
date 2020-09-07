import librosa
import torch

from scipy.io.wavfile import write


# load wav data from file
def load_wav(fn):
    data, sr = librosa.load(fn)
    data = 0.95 * librosa.util.normalize(data)
    return torch.from_numpy(data).float(), sr


# load raw wav and mel-spectrogram from fn by using official melgan's converter
def get_wav_mel(fn, to_mel=None):
    if to_mel is None:
        to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, sr = librosa.load(fn)
    wav = 0.95 * librosa.util.normalize(wav)
    mel = to_mel(torch.tensor(wav, dtype=torch.float).unsqueeze(0)).squeeze(0)
    return wav, mel.cpu()


# save sample
def save_sample(file_path, audio, sr=22050):
    audio = (audio * 32768).astype("int16")
    write(file_path, sr, audio)
