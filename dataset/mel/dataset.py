from ..base import DatasetBase
from utils import get_wav


class MelDataset(DatasetBase):
    def __getitem__(self, idx):
        wav = self.data[idx]
        mel = self._to_mel(wav)
        return mel, mel.clone()


class MelFromFileDataset(DatasetBase):
    def __getitem__(self, idx):
        fn = self.data[idx]
        _wav = get_wav(fn)
        mel = self._to_mel(_wav)
        return mel, mel.clone()
