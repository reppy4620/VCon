import librosa
import torch

from dataset import VConDataModule
from utils import (
    get_config, get_wav_mel, save_sample, model_from_config
)


def config_test():
    config1 = get_config('configs/autovc_vqvae.yaml')
    config2 = get_config('configs/quartz.yaml')
    print(config1)
    print(config2)


def autovc_base_vqvae_test():
    fn = ''
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc_vqvae.yaml')
    wav, mel = get_wav_mel(fn, to_mel)
    model = model_from_config(params)
    out, _ = model(wav, mel.cpu().unsqueeze(0))
    print(mel.size(), out.size())


def quartz_test():
    fn = ''
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/quartz.yaml')
    wav, mel = get_wav_mel(fn, vocoder)
    model = model_from_config(params)
    out = model([wav], mel.unsqueeze(0))
    print(out.size())


def autovc_test():
    fn = ''
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc.yaml')
    wav, mel = get_wav_mel(fn, to_mel)
    model = model_from_config(params)
    out_dec, out_ptnt, enc_real, enc_fake = model([wav], mel[None, :, :64])
    print(out_dec.size(), out_ptnt.size(), enc_real.size(), enc_fake.size())


def load_wav_mel_test():
    fn = ''
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, _, mel = get_wav_mel(fn, to_mel)
    print(wav.size(), mel.size())


def melgan_test():
    fn = ''
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc_vqvae.yaml')
    wav, mel = get_wav_mel(fn, vocoder)
    print(mel.min(), mel.max())
    mel_exp = torch.expm1(mel)
    print(mel_exp.min(), mel_exp.max())
    mel = torch.log1p(mel_exp)
    print(mel.min(), mel.max())
    with torch.no_grad():
        audio = vocoder.inverse(mel.unsqueeze(0))[0]
    audio = audio.cpu().detach().numpy()
    save_sample('./test.wav', audio)


def data_loader_test():
    params = get_config('configs/quartz.yaml')
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    # params.data_dir = 'D:/dataset/libritts/LibriTTS/processed'
    params.data_dir = 'D:/dataset/vctk/processed'
    dm = VConDataModule(params)
    dm.setup()
    train_loader = dm.train_dataloader()
    res = None
    wavs = None
    for data in train_loader:
        wav, mel = data
        res = vocoder.inverse(mel[:10, :, :]).cpu().detach().numpy()
        wavs = wav
        break
    [save_sample(f'./test_wav/test-{i}.wav', res[i]) for i in range(res.shape[0])]


if __name__ == '__main__':
    autovc_test()
