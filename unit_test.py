import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
from librosa.display import specshow
from scipy.io.wavfile import write

from dataset import VConDataModule
from nn import VCModel
from utils import get_config, get_wav_mel, load_data


def config_test():
    config = get_config('config.yaml')
    print(config.batch_size, config.encoder.n_layers)


def model_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('config.yaml')
    wav, _, mel = get_wav_mel(fn, to_mel)
    model = VCModel(params)
    out, _ = model(wav, mel.cpu().unsqueeze(0))
    print(mel.size(), out.size())


def load_wav_file_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, _, mel = get_wav_mel(fn, to_mel)

    plt.plot(wav)
    plt.show()

    specshow(np.log1p(mel.T))
    plt.show()

    wav = librosa.feature.inverse.mel_to_audio(mel.T)
    plt.plot(wav)
    plt.show()


def load_wav_mel_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, _, mel = get_wav_mel(fn, to_mel)
    print(wav.size(), mel.size())


def melgan_test():
    fn = 'D:/dataset/seiyu/fujitou_normal/fujitou_normal_001.wav'
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('config.yaml')
    _, _, mel = get_wav_mel(fn, vocoder)
    with torch.no_grad():
        audio = vocoder.inverse(mel.unsqueeze(0))[0]
    audio = audio.cpu().detach().numpy()
    write('./test.wav', params.sampling_rate, audio)


def data_loader_test():
    params = get_config('config.yaml')
    dm = VConDataModule(params)
    dm.setup()
    train_loader = dm.train_dataloader()
    for data in train_loader:
        wav, wav_pt, mel = data
        print(wav.shape, wav_pt.size(), mel.size())
        break


def data_load_test():
    params = get_config('config.yaml')
    a = load_data(pathlib.Path(params.data_dir))
    print(len(a))


if __name__ == '__main__':
    data_load_test()
