import pytorch_lightning as pl
import torch.nn.functional as F

from .model import VQVCModel
from optim import RAdam
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
            time_drop_width=3,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx):
        _, mel = batch

        m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)

        dec, enc, quant_diff = self.model(_, m)

        l_recon = F.l1_loss(dec, mel)

        loss = l_recon + quant_diff

        result = pl.TrainResult(loss)
        result.log_dict({
            'loss': loss,
            'l_recon': l_recon,
            'quant_diff': quant_diff,
        }, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        _, mel = batch

        dec, enc, quant_diff = self.model(_, mel)

        l_recon = F.l1_loss(dec, mel)

        loss = l_recon + quant_diff

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_quant_diff': quant_diff,
        }, prog_bar=True)

        return result

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)
