from ..base import DatasetBase


class Wav2VecMelDataset(DatasetBase):
    def __getitem__(self, idx):
        z, wav = self.data[idx]
        mel = self._to_mel(wav)
        return z, mel
