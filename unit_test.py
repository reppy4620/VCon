import pathlib

import librosa
import torch
from scipy.io.wavfile import write

from dataset import VConDataModule
from utils import (
    get_config, get_wav_mel, load_data,
    save_sample, trim_long_silences, model_from_config
)


def config_test():
    config1 = get_config('configs/autovc_vqvae.yaml')
    config2 = get_config('configs/quartz.yaml')
    print(config1)
    print(config2)


def autovc_base_vqvae_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc_vqvae.yaml')
    wav, _, mel = get_wav_mel(fn, to_mel)
    model = model_from_config(params)
    out, _ = model(wav, mel.cpu().unsqueeze(0))
    print(mel.size(), out.size())


def quartz_test():
    fn = 'D:/dataset/seiyu/fujitou_normal/fujitou_normal_001.wav'
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/quartz.yaml')
    wav, mel = get_wav_mel(fn, vocoder)
    model = model_from_config(params)
    out = model([wav], mel.unsqueeze(0))
    print(out.size())


def load_wav_mel_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, _, mel = get_wav_mel(fn, to_mel)
    print(wav.size(), mel.size())


def melgan_test():
    fn = 'D:/dataset/seiyu/fujitou_normal/fujitou_normal_001.wav'
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc_vqvae.yaml')
    _, mel = get_wav_mel(fn, vocoder)
    with torch.no_grad():
        audio = vocoder.inverse(mel.unsqueeze(0))[0]
    audio = audio.cpu().detach().numpy()
    audio = (audio * 32768).astype('int16')
    write('./test.wav', params.sampling_rate, audio)


def data_loader_test():
    params = get_config('configs/autovc_vqvae.yaml')
    dm = VConDataModule(params)
    dm.setup()
    train_loader = dm.train_dataloader()
    for data in train_loader:
        wav, wav_pt, mel = data
        print(wav.shape, wav_pt.size(), mel.size())
        break


def data_load_test():
    params = get_config('configs/autovc_vqvae.yaml')
    a = load_data(pathlib.Path(params.data_dir))
    print(len(a))


def trim_test():
    fn = 'D:/dataset/vctk/wav/p226/p226_001.wav'
    wav, sr = librosa.load(fn)
    wav = trim_long_silences(wav)
    save_sample('test2.wav', wav, sr)


if __name__ == '__main__':
    quartz_test()
