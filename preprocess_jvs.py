import argparse
import torch

from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import get_wav


def process_one(fn):
    wav = get_wav(fn)
    return wav


def process_dir(data_dir, output_dir):
    wav_files = list(data_dir.glob('parallel100/wav24kHz16bit/*.wav')) + list(data_dir.glob('nonpara30/wav24kHz16bit/*.wav'))
    if len(wav_files) == 0:
        return
    data = [process_one(fn) for fn in wav_files]
    torch.save(data, output_dir / str(data_dir.name + '.dat'))


def preprocess(dataset_dir: Path, output_dir: Path):

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fns = list(dataset_dir.glob('jvs*'))
    Parallel(n_jobs=-1)(delayed(process_dir)(d, output_dir) for d in tqdm(fns, total=len(fns)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    preprocess(dataset_dir, output_dir)
