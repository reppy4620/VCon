import pytorch_lightning as pl
import torch.nn.functional as F

from optim import RAdam
from transforms import SpecAugmentation
from utils import AttributeDict
from .model import QuartzModel


class QuartzModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = QuartzModel(params)

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

        out_post, enc_real, enc_fake = self.model(wav, m)

        l_recon = F.mse_loss(out_post, mel)
        l_cont = F.l1_loss(enc_fake, enc_real)

        loss = l_recon + l_cont

        self.log_dict({
            'train_loss': loss,
            'l_recon': l_recon,
            'l_cont': l_cont
        }, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav, mel = batch

        out_post, enc_real, enc_fake = self.model(wav, mel)

        l_recon = F.mse_loss(out_post, mel)
        l_cont = F.l1_loss(enc_fake, enc_real)

        loss = l_recon + l_cont

        self.log_dict({
            'val_loss': loss,
            'val_l_recon': l_recon,
            'val_l_cont': l_cont
        }, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return RAdam(self.model.parameters(), self.hparams.optimizer.lr)
