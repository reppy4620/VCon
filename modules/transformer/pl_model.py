import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief

from utils import AttributeDict
from .model import TransformerModel


class TransformerModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = TransformerModel(params)

    def forward(self, spec_src, spec_tgt):
        return self.model.inference(spec_src, spec_tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        out, q_loss = self.model(src, tgt)

        recon_loss = F.l1_loss(out, src)
        loss = recon_loss + 0.1 * q_loss

        self.log_dict({
            'loss': loss,
            'r_loss': recon_loss,
            'q_loss': q_loss
        }, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        out, q_loss = self.model(src, tgt)

        recon_loss = F.l1_loss(out, src)
        loss = recon_loss + 0.1 * q_loss

        self.log_dict({
            'val_loss': loss,
            'val_r_loss': recon_loss,
            'val_q_loss': q_loss,
            'step': self.global_step
        }, prog_bar=True)
        return out[0]

    def validation_step_end(self, val_outputs):

        if self.current_epoch % 10 == 0:
            if isinstance(val_outputs, torch.Tensor):
                mel = val_outputs.unsqueeze(0)
            else:
                mel = val_outputs[0].unsqueeze(0)
            wav = self.model.inverse(mel)
            self.logger.experiment.add_audio('sample', wav, global_step=self.global_step, sample_rate=22050)
            self.logger.experiment.add_image('mel', mel[0], global_step=self.global_step, dataformats='HW')

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=0.9, nesterov=True)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, eta_min=1e-6)
        # optimizer = AdaBelief(
        #     params=self.model.parameters(),
        #     lr=self.hparams.optimizer.lr,
        #     betas=(0.9, 0.999),
        #     eps=1e-16,
        #     weight_decouple=True,
        #     rectify=True,
        #     fixed_decay=False,
        #     amsgrad=False
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, eta_min=1e-6)
        # return [optimizer], [scheduler]
        return AdaBelief(
            params=self.model.parameters(),
            lr=self.hparams.optimizer.lr,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decouple=True,
            rectify=True,
            fixed_decay=False,
            amsgrad=False
        )
