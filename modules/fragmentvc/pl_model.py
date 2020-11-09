import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief

from .model import FragmentVCModel
from transforms import SpecAugmentation
from utils import AttributeDict


class FragmentVCModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)

        self.hparams = params

        self.model = FragmentVCModel(params)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=3,
            time_stripes_num=2,
            freq_drop_width=3,
            freq_stripes_num=2
        )

    def forward(self, spec_src, spec_tgt):
        return self.model.inference(spec_src, spec_tgt)

    def training_step(self, batch, batch_idx):
        src, ref, tgt, src_mask, tgt_mask, overlap_lens = batch

        ref_h = self.spec_augmenter(ref)

        out, _ = self.model(src, ref_h, src_mask, tgt_mask)

        loss = self._loss(out, tgt, overlap_lens)

        self.log('loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, ref, tgt, src_mask, tgt_mask, overlap_lens = batch

        out, _ = self.model(src, ref, src_mask, tgt_mask)

        loss = self._loss(out, tgt, overlap_lens)

        self.log('val_loss', loss, prog_bar=True)
        return out[0]

    def validation_step_end(self, val_outputs):
        if self.global_step % 1000 == 0:
            if isinstance(val_outputs, torch.Tensor):
                mel = val_outputs.unsqueeze(0)
            else:
                mel = val_outputs[0].unsqueeze(0)
            wav = self.model.vocoder.inverse(mel).detach().cpu()
            self.logger.experiment.add_audio('sample', wav, global_step=self.global_step, sample_rate=22050)
            self.logger.experiment.add_image('mel', mel[0], global_step=self.global_step, dataformats='HW')

    @staticmethod
    def _loss(src, tgt, overlap_lens):
        loss = 0
        for s, t, l in zip(src.unbind(), tgt.unbind(), overlap_lens):
            loss += F.l1_loss(s[:, :l], t[:, :l])
        return loss

    def configure_optimizers(self):
        # for transformer
        return AdaBelief(
            params=self.model.parameters(),
            lr=self.hparams.optimizer.lr,
            eps=1e-16,
            weight_decay=1e-4,
            weight_decouple=True,
            rectify=True,
            fixed_decay=False,
            amsgrad=False
        )
