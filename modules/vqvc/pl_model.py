import pytorch_lightning as pl
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief

from .model import VQVCModel
from transforms import SpecAugmentation
from utils import AttributeDict


class VQVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = VQVCModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=6,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, src_path: str, tgt_path: str):
        return self.model.inference(src_path, tgt_path)

    def training_step(self, batch, batch_idx):
        wavs, mels = batch

        m = self.spec_augmenter(mels.unsqueeze(1)).squeeze(1)

        dec, quant_diff = self.model(wavs, m)

        l_recon = F.smooth_l1_loss(dec, mels)

        loss = l_recon + quant_diff
        self.log_dict({
            'loss': loss,
            'l_recon': l_recon,
            'quant_diff': quant_diff,
        }, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wavs, mels = batch

        dec, quant_diff = self.model(wavs, mels)

        l_recon = F.smooth_l1_loss(dec, mels)

        loss = l_recon + quant_diff
        self.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_quant_diff': quant_diff,
        }, prog_bar=True)

    def configure_optimizers(self):
        return AdaBelief(self.model.parameters(), self.hparams.optimizer.lr)
