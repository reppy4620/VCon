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

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=6,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, spec_src, spec_tgt):
        return self.model.inference(spec_src, spec_tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        src_h = self.spec_augmenter(src.unsqueeze(1)).squeeze(1)
        tgt_h = self.spec_augmenter(src.unsqueeze(1)).squeeze(1)

        out = self.model(src_h, tgt_h)

        loss = F.smooth_l1_loss(out, src_h)

        self.log('loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        out = self.model(src, tgt)

        loss = F.smooth_l1_loss(out, src)

        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return AdaBelief(self.model.parameters(), self.hparams.optimizer.lr)
