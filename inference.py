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
