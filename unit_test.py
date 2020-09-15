import torch

from dataset import VConDataModule
from utils import (
    get_config, get_wav_mel, save_sample, model_from_config,
    normalize, denormalize
)


def config_test():
    config1 = get_config('configs/autovc_vqvae.yaml')
    config2 = get_config('configs/quartz.yaml')
    print(config1)
    print(config2)


def autovc_base_vqvae_test():
    fn = 'D:/dataset/seiyu/fujitou_normal/fujitou_normal_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc_vqvae.yaml')
    wav, mel = get_wav_mel(fn, to_mel)
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


def autovc_test():
    torch.backends.cudnn.enabled = False
    fn = 'D:/dataset/seiyu/fujitou_normal/fujitou_normal_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    params = get_config('configs/autovc.yaml')
    wav, mel = get_wav_mel(fn, to_mel)
    model = model_from_config(params).cuda()
    out_dec, out_ptnt, enc_real, enc_fake = model([wav], mel[None, :, :64].float().cuda())
    print(out_dec.size(), out_ptnt.size(), enc_real.size(), enc_fake.size())


def load_wav_mel_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, _, mel = get_wav_mel(fn, to_mel)
    print(wav.size(), mel.size())


def melgan_test():
    fn = 'D:/dataset/vctk/wav/p225/p225_001.wav'
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, mel = get_wav_mel(fn, vocoder)
    mel = normalize(mel)
    mel = denormalize(mel)
    print(mel.min(), mel.max(), mel.mean())
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
        print(mel.min(), mel.max(), mel.mean())
        res = vocoder.inverse(mel[:10, :, :]).cpu().detach().numpy()
        wavs = wav
        break
    [save_sample(f'./test_wav/test-{i}.wav', res[i]) for i in range(res.shape[0])]


def normalize_test():
    fn = 'D:/dataset/vctk/wav/p225/p225_001.wav'
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, mel = get_wav_mel(fn, vocoder)
    m_norm = normalize(mel)
    m_denorm = denormalize(m_norm)


if __name__ == '__main__':
    autovc_test()
