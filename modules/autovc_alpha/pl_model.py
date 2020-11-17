import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief

from utils import AttributeDict
from .model import AutoVCAlphaModel


class AutoVCAlphaModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = AutoVCAlphaModel(params)

    def forward(self, src_path: str, tgt_path: str):
        return self.model.inference(src_path, tgt_path)

    def training_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon, q_loss = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont + q_loss
        self.log_dict({
            'train_loss': loss,
            'l_recon': l_recon,
            'l_recon0': l_recon0,
            'l_cont': l_cont,
            'l_vq': q_loss
        }, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon, q_loss = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont + q_loss
        self.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_l_recon0': l_recon0,
            'val_l_cont': l_cont,
            'val_l_vq': q_loss,
            'step': self.global_step
        }, prog_bar=True)
        return out_psnt[0]

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
        return AdaBelief(
            params=self.model.parameters(),
            lr=self.hparams.optimizer.lr,
            eps=1e-12,
            weight_decay=1.2e-6,
            weight_decouple=False,
            rectify=False,
            fixed_decay=False,
            amsgrad=False
        )
