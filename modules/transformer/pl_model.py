import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief

from .model import TransformerModel
from transforms import SpecAugmentation
from utils import AttributeDict


class TransformerModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = TransformerModel(params)

        # self.spec_augmenter = SpecAugmentation(
        #     time_drop_width=3,
        #     time_stripes_num=2,
        #     freq_drop_width=3,
        #     freq_stripes_num=2
        # )

    def forward(self, spec_src, spec_tgt):
        return self.model.inference(spec_src, spec_tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # src_h = self.spec_augmenter(src)
        # tgt_h = self.spec_augmenter(tgt)

        # out, q_loss = self.model(src_h, tgt_h)
        out, q_loss = self.model(src, tgt)

        recon_loss = F.l1_loss(out, src)
        loss = 100 * recon_loss + q_loss

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
        loss = 100 * recon_loss + q_loss

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
        optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        return [optimizer], [scheduler]
