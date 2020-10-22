import torch

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


# load data from preprocessed files
def load_data(data_dir: Path, ratio: float, parallel: bool = False):
    # parallel processing has a problem.

    print('Load training data')
    fns = list(data_dir.glob('*.dat'))[:10]
    if parallel:
        data = Parallel(n_jobs=12)(delayed(torch.load)(str(d)) for d in tqdm(fns, total=len(fns)))
    else:
        data = [torch.load(str(d)) for d in tqdm(fns, total=len(fns))]
    num_of_train = int(len(data) * ratio)
    train, valid = data[:num_of_train], data[num_of_train:]
    return sum(train, list()), sum(valid, list())


def load_fns(data_dir: Path, ratio: float):
    print('Load training fns')
    dirs = list(data_dir.glob(''))
