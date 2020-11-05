import pytorch_lightning as pl
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief

from transforms import SpecAugmentation
from utils import AttributeDict
from .model import AutoVCModel


class AutoVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = AutoVCModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=3,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, src_path: str, tgt_path: str):
        return self.model.inference(src_path, tgt_path)

    def training_step(self, batch, batch_idx):
        wav, mel = batch

        # m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)

        out_dec, out_psnt, c_real, c_recon = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont
        self.log_dict({
            'train_loss': loss,
            'l_recon': l_recon,
            'l_recon0': l_recon0,
            'l_cont': l_cont
        }, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont
        self.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_l_recon0': l_recon0,
            'val_l_cont': l_cont
        }, prog_bar=True)

    def configure_optimizers(self):
        return AdaBelief(self.model.parameters(), self.hparams.optimizer.lr)
