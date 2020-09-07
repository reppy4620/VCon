import torch

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


# load data from preprocessed files
def load_data(data_dir: Path):
    print('Load training data')
    fns = list(data_dir.glob('*.dat'))
    # data = Parallel(n_jobs=-1)(delayed(torch.load)(str(d)) for d in tqdm(fns, total=len(fns)))
    data = [torch.load(str(d)) for d in tqdm(fns, total=len(fns))]
    return sum(data, list())
