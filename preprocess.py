import argparse
import torch

from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import get_wav_mel


def process_one(fn, to_mel):
    wav, mel = get_wav_mel(fn, to_mel)
    return wav, mel


def process_dir(data_dir, output_dir):
    print(f'Start process : {str(data_dir)}')
    wav_files = data_dir.glob('*.wav')
    _wav_to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    data = list()
    for fn in wav_files:
        data.append(process_one(fn, _wav_to_mel))
    torch.save(data, output_dir / str(data_dir.name + '.dat'))


def preprocess(dataset_dir: Path, output_dir: Path):

    if not output_dir.exists():
        output_dir.mkdir()

    fns = list(dataset_dir.glob('*'))
    Parallel(n_jobs=-1)(delayed(process_dir)(d, output_dir) for d in tqdm(fns, total=len(fns)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    preprocess(dataset_dir, output_dir)
