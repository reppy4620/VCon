import torch
import argparse

from utils import get_config, get_wav_mel, save_sample, module_from_config


def inference(src_fn, tgt_fn, model):
    src_wav, src_mel = get_wav_mel(src_fn)
    tgt_wav, _ = get_wav_mel(tgt_fn)

    wav = model(src_wav, tgt_wav, src_mel)

    return wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--out_path', type=str, default='./out.wav')
    args = parser.parse_args()

    args.ckpt_path = 'D:/models/vcon/quartz/libritts/vc_epoch=353-val_loss=0.02.ckpt'

    params = get_config(args.config_path)

    print('Build model')
    model = module_from_config(params)
    model.load_from_checkpoint(args.ckpt_path)
    model.freeze()

    print('Inference')
    wav = inference(args.src_path, args.tgt_path, model)

    print('Saving')
    save_sample(args.out_path, wav)
    print('End')
