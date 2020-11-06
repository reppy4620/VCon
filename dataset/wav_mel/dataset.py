from ..base import DatasetBase
from utils import get_wav


# Simple dataset
class WavMelDataset(DatasetBase):
    def __getitem__(self, idx):
        wav = self.data[idx]
        mel = self._to_mel(wav)
        return wav, mel


class WavMelFromFileDataset(DatasetBase):
    def __getitem__(self, idx):
        fn = self.data[idx]
        wav = get_wav(fn)
        mel = self._to_mel(wav)
        return wav, mel
