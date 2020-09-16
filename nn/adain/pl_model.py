import pytorch_lightning as pl

import torch.nn.functional as F

from .model import AdaINVCModel
from utils import AttributeDict
from transforms import SpecAugmentation
from optim import RAdam


class AdaINVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = AdaINVCModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=8,
            time_stripes_num=2,
            freq_drop_width=4,
            freq_stripes_num=2
        )

    def forward(self, raw_src, raw_tgt, spec_src):
        return self.model.inference(raw_src, raw_tgt, spec_src)

    def training_step(self, batch, batch_idx):
        wav, mel = batch

        m = self.spec_augmenter(mel.unsqueeze(1)).squeeze(1)

        out = self.model(wav, m)

        l_recon = F.l1_loss(out, mel)

        result = pl.TrainResult(l_recon)
        result.log_dict({
            'train_loss': l_recon,
        }, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out = self.model(wav, mel)

        l_recon = F.l1_loss(out, mel)

        result = pl.EvalResult(checkpoint_on=l_recon)
        result.log_dict({
            'val_loss': l_recon,
        }, prog_bar=True)
        return result

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)
