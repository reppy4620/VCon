import argparse
import pathlib

import torch.nn.functional as F

from utils import get_config, get_wav_mel, save_sample, module_from_config, normalize


# for AutoVC model
def _preprocess(mel, freq=32):
    t_dim = mel.size(-1)
    mod_val = t_dim % freq
    if mod_val != 0:
        if mod_val < t_dim / 2:
            mel = mel[:, :t_dim-mod_val]
        else:
            pad_length = t_dim + freq - mod_val
            mel = F.pad(mel[None, :, :], [0, pad_length], value=-5).squeeze(0)
    mel = normalize(mel)
    return mel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./outputs/autovc')
    args = parser.parse_args()

    params = get_config(args.config_path)

    output_dir = pathlib.Path(args.output_dir) / args.ckpt_path.split('/')[-2]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print('Load wav')
    src_wav, src_mel = get_wav_mel(args.src_path)
    tgt_wav, _ = get_wav_mel(args.tgt_path)

    if 'autovc' in params.exp_name:
        src_mel = _preprocess(src_mel, freq=params.model.freq)
    elif 'vqvc' in params.exp_name:
        src_mel = _preprocess(src_mel, freq=4)

    print('Build model')
    model = module_from_config(params)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.freeze()

    print('Inference')
    wav = model(src_wav, tgt_wav, src_mel)

    print('Saving')
    save_sample(str(output_dir / 'src.wav'), src_wav)
    save_sample(str(output_dir / 'tgt.wav'), tgt_wav)
    save_sample(str(output_dir / 'gen.wav'), wav)
    print('End')
