import torch
import torch.optim as optim

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from nn import VCModel


class VCModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = VCModel(hparams)

    def forward(self, raw_src, raw_tgt, spec_src):
        c_src = self.speaker_encoder.embed_utterance(raw_src)
        c_tgt = self.speaker_encoder.embed_utterance(raw_tgt)
        out, _ = self.vq_vae(spec_src, c_src, c_tgt)
        return out

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_end(self, outputs):
        pass

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
