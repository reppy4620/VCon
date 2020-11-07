import argparse
from enum import IntEnum
from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from transforms import Wav2Mel
from utils import get_wav, get_world_features, get_wav2vec_features
from utils import load_pretrained_wav2vec


# I don't know proper implementation for enum
class DataType(IntEnum):
    NORMAL = 1
    WORLD = 2
    WAV2VEC = 3

    type_dict = {
        'normal': NORMAL,
        'world': WORLD,
        'wav2vec': WAV2VEC
    }

    @classmethod
    def from_str(cls, s):
        return cls.type_dict[s]


def process_one(fn, data_type, wav2vec=None, wav2mel=None):
    if data_type == DataType.NORMAL:
        wav = get_wav(fn)
        return wav
    elif data_type == DataType.WORLD:
        f0, sp, ap = get_world_features(fn)
        return f0, sp, ap
    elif data_type == DataType.WAV2VEC:
        feat, mel = get_wav2vec_features(fn, wav2vec, wav2mel)
        return feat, mel
    else:
        raise ValueError('Invalid value: type of data_type must be DataType')


def process_dir(data_dir, output_dir, data_type, wav2vec_path=None):
    wav_files = list(data_dir.glob('parallel100/wav24kHz16bit/*.wav')) + list(data_dir.glob('nonpara30/wav24kHz16bit/*.wav'))
    if len(wav_files) == 0:
        return
    wav2vec, wav2mel = None, None
    if data_type == DataType.WAV2VEC:
        wav2vec = load_pretrained_wav2vec(wav2vec_path)
        wav2mel = Wav2Mel()
    data = [process_one(fn, data_type, wav2vec, wav2mel) for fn in wav_files]
    torch.save(data, output_dir / str(data_dir.name + '.dat'))


def preprocess(dataset_dir: Path, output_dir: Path, data_type: DataType, wav2vec_path: str = None):

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fns = list(dataset_dir.glob('jvs*'))
    Parallel(n_jobs=8)(
        delayed(process_dir)(d, output_dir, data_type, wav2vec_path) for d in tqdm(fns, total=len(fns))
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_type', type=str, choices=['normal', 'world', 'wav2vec'], required=True)
    args = parser.parse_args()

    assert args.world and args.wav2vec

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    data_type = DataType.from_str(args.data_type)

    preprocess(dataset_dir, output_dir, data_type)
