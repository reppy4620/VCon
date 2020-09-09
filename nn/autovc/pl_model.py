import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from optim import RAdam
from utils import AttributeDict
from .model import AutoVCModel


class AutoVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = AutoVCModel(params)

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont

        log = {'loss': loss, 'l_recon': l_recon, 'l_recon0': l_recon0, 'l_cont': l_cont}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont

        return {
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_l_recon0': l_recon0,
            'val_l_cont': l_cont
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_l_recon = torch.stack([x['val_l_recon'] for x in outputs]).mean()
        avg_l_recon0 = torch.stack([x['val_l_recon0'] for x in outputs]).mean()
        avg_l_cont = torch.stack([x['val_l_cont'] for x in outputs]).mean()
        log = {
            'val_loss': avg_loss,
            'val_l_recon': avg_l_recon,
            'val_l_recon0': avg_l_recon0,
            'val_l_cont': avg_l_cont
        }
        return {'val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)
