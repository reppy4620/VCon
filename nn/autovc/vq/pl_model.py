import pytorch_lightning as pl
import torch.nn.functional as F

from .model import VQAutoVCModel
from optim import RAdam
from transforms import SpecAugmentation
from utils import AttributeDict


class VQAutoVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = VQAutoVCModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=3,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx):
        wav, mel = batch

        m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)

        out_dec, out_psnt, c_real, c_recon, l_vq = self.model(wav, m)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont + l_vq

        result = pl.TrainResult(loss)
        result.log_dict({
            'loss': loss,
            'l_recon': l_recon,
            'l_recon0': l_recon0,
            'l_cont': l_cont,
            'l_vq': l_vq
        }, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out_dec, out_psnt, c_real, c_recon, l_vq = self.model(wav, mel)

        l_recon0 = F.mse_loss(out_dec, mel)
        l_recon = F.mse_loss(out_psnt, mel)
        l_cont = F.l1_loss(c_real, c_recon)

        loss = l_recon + l_recon0 + l_cont + l_vq

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_l_recon0': l_recon0,
            'val_l_cont': l_cont,
            'val_l_vq': l_vq
        }, prog_bar=True)

        return result

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)