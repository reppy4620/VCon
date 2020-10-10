import pytorch_lightning as pl

import torch.nn.functional as F

from .model import AdaINVCModel
from .network import Discriminator
from utils import AttributeDict
from transforms import SpecAugmentation
from optim import RAdam


class AdaINGANModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = AdaINVCModel(params)
        self.discriminator = Discriminator(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=8,
            time_stripes_num=2,
            freq_drop_width=4,
            freq_stripes_num=2
        )

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx, optimizer_idx):
        wav, mel = batch

        m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)

        fake = self.model(wav, m)

        if optimizer_idx == 0:
            pred_real = self.discriminator(mel)
            loss_d_real = F.relu(1.0 - pred_real).mean()

            pred_fake = self.discriminator(fake.detach())
            loss_d_fake = F.relu(1.0 + pred_fake).mean()

            loss_d = loss_d_real + loss_d_fake

            self.log_dict({
                'loss_d': loss_d,
                'loss_d_real': loss_d_real,
                'loss_d_fake': loss_d_fake
            })
            return loss_d

        elif optimizer_idx == 1:
            g_pred_fake = self.discriminator(fake)

            loss_g_gan = -g_pred_fake.mean()
            loss_recon = F.l1_loss(fake, mel)

            loss_g = loss_g_gan + loss_recon * 10

            self.log_dict({
                'loss_g': loss_g,
                'loss_g_gan': loss_g_gan,
                'loss_recon': loss_recon,
            }, on_epoch=True)
            return loss_g

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out = self.model(wav, mel)

        l_recon = F.l1_loss(out, mel)

        self.log_dict({
            'val_loss': l_recon,
        }, prog_bar=True)
        return l_recon

    def configure_optimizers(self):
        optD = RAdam(
            self.discriminator.parameters(),
            self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.b1, self.hparams.optimizer.b2)
        )
        optG = RAdam(
            self.model.parameters(),
            self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.b1, self.hparams.optimizer.b2)
        )
        return [optD, optG], []
