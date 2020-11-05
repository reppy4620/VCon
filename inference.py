import argparse
import pathlib

from utils import get_config, get_wav, save_sample, module_from_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./outputs/vqvc')
    args = parser.parse_args()

    params = get_config(args.config_path)

    output_dir = pathlib.Path(args.output_dir) / args.ckpt_path.split('/')[-2]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print('Build model')
    model = module_from_config(params)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.freeze()

    print('Inference')
    wav = model(args.src_path, args.tgt_path)

    print('Saving')
    src_wav, tgt_wav = get_wav(args.src_path), get_wav(args.tgt_path)
    save_sample(str(output_dir / 'src.wav'), src_wav)
    save_sample(str(output_dir / 'tgt.wav'), tgt_wav)
    save_sample(str(output_dir / 'gen.wav'), wav)
    print('End')
