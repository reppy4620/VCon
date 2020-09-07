import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from transforms import SpecAugmentation
from nn import VCModel
from optim import RAdam


class VCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.params = params

        self.model = VCModel(params)

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
        mel = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)
        out, vq_loss = self.model(wav, mel)
        l_recon = F.mse_loss(mel, out)
        loss = l_recon + vq_loss

        log = {'loss': loss, 'l_recon': l_recon, 'diff': vq_loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        wav, mel = batch
        out, vq_loss = self.model(wav, mel)
        l_recon = F.mse_loss(mel, out)
        loss = l_recon + vq_loss

        return {'val_loss': loss, 'val_l_recon': l_recon, 'val_vq_loss': vq_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        avg_l_recon = torch.stack([x['val_l_recon'] for x in outputs]).sum()
        avg_vq_loss = torch.stack([x['val_vq_loss'] for x in outputs]).sum()
        log = {'val_loss': avg_loss, 'avg_l_recon': avg_l_recon, 'avg_vq_loss': avg_vq_loss}
        return {'val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.params.optimizer.lr)
