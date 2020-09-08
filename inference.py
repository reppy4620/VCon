import torch
import argparse

from utils import get_config, get_wav_mel, save_sample, module_from_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--out_path', type=str, default='./out.wav')
    args = parser.parse_args()

    # args.src_path = 'D:/dataset/jvs/jvs001/nonpara30/wav24kHz16bit/BASIC5000_0025.wav'
    # args.tgt_path = 'D:/dataset/jvs/jvs004/nonpara30/wav24kHz16bit/VOICEACTRESS100_003.wav'
    args.src_path = 'D:/dataset/vctk/wav/p226/p226_002.wav'
    args.src_path = 'D:/dataset/vctk/wav/p225/p225_001.wav'
    args.config_path = 'configs/quartz.yaml'
    args.ckpt_path = 'D:/models/'

    params = get_config(args.config_path)

    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    print('Load wav')
    src_wav, src_mel = get_wav_mel(args.src_path)
    tgt_wav, _ = get_wav_mel(args.tgt_path)

    print('Build model')
    model = module_from_config(params)
    model.load_from_checkpoint(args.ckpt_path)
    model.freeze()

    print('Inference')
    wav = model(src_wav, tgt_wav, src_mel)

    print('Saving')
    save_sample(args.out_path, wav)
    print('End')
