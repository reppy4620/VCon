import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transforms import SpecAugmentation
from .model import QuartzModel
from optim import RAdam
from utils import Map


class QuartzModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, Map):
            params = Map(params)

        self.hparams = params

        self.model = QuartzModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=32,
            time_stripes_num=2,
            freq_stripes_num=8,
            freq_drop_width=2
        )

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx):
        wav, mel = batch
        m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)
        out = self.model(wav, m)
        l_recon = F.mse_loss(mel, out)

        log = {'loss': l_recon}
        return {'loss': l_recon, 'log': log}

    def validation_step(self, batch, batch_idx):
        wav, mel = batch
        out = self.model(wav, mel)
        l_recon = F.mse_loss(mel, out)

        return {'val_loss': l_recon}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        log = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)
