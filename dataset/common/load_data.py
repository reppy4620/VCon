import torch

from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict


# load data from preprocessed files
def load_data(data_dir: Path, ratio: float):
    fns = list(data_dir.glob('*.dat'))
    data = [torch.load(str(d)) for d in tqdm(fns, total=len(fns))]
    num_of_train = int(len(data) * ratio)
    train, valid = data[:num_of_train], data[num_of_train:]
    return sum(train, list()), sum(valid, list())


def load_fns(root: Path, ratio: float):
    def _fns_from_dir(data_dir: Path):
        wav_files = list(data_dir.glob('parallel100/wav24kHz16bit/*.wav')) + \
                    list(data_dir.glob('nonpara30/wav24kHz16bit/*.wav'))
        return wav_files
    data_dirs = list(root.glob('jvs*'))
    data = [_fns_from_dir(d) for d in tqdm(data_dirs, total=len(data_dirs))]
    num_of_train = int(len(data) * ratio)
    train, valid = data[:num_of_train], data[num_of_train:]
    return sum(train, list()), sum(valid, list())


# load data from preprocessed files
def load_data_with_indices(data_dir: Path, ratio: float):
    fns = list(data_dir.glob('*.dat'))
    cnt = 0
    data = list()
    indices = OrderedDict()
    for i, fn in tqdm(enumerate(fns), total=len(fns)):
        d = torch.load(str(fn))
        data.append(list(zip([i] * len(d), d)))
        indices[i] = list(range(cnt, cnt+len(d)))
        cnt += len(d)
    num_of_train = int(len(data) * ratio)
    train, valid = data[:num_of_train], data[num_of_train:]
    indices = list(indices.items())
    train_i, valid_i = OrderedDict(indices[:num_of_train]), OrderedDict(indices[num_of_train:])
    return sum(train, list()), sum(valid, list()), train_i, valid_i


# load data from preprocessed files
def load_fns_with_indices(root: Path, ratio: float):
    def _fns_from_dir(data_dir: Path):
        wav_files = list(data_dir.glob('parallel100/wav24kHz16bit/*.wav')) + \
                    list(data_dir.glob('nonpara30/wav24kHz16bit/*.wav'))
        return wav_files
    data_dirs = list(root.glob('jvs*'))
    cnt = 0
    data = list()
    indices = OrderedDict()
    for i, d in tqdm(enumerate(data_dirs), total=len(data_dirs)):
        fns = _fns_from_dir(d)
        data.append(list(zip([i] * len(fns), fns)))
        indices[i] = list(range(cnt, cnt+len(fns)))
        cnt += len(fns)
    num_of_train = int(len(data) * ratio)
    train, valid = data[:num_of_train], data[num_of_train:]
    indices = list(indices.items())
    train_i, valid_i = OrderedDict(indices[:num_of_train]), OrderedDict(indices[num_of_train:])
    return sum(train, list()), sum(valid, list()), train_i, valid_i
