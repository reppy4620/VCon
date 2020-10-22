import torch.nn as nn


class ModelMixin(nn.Module):

    # for training
    def forward(self, raw, spec):
        raise NotImplementedError

    # for inference
    def inference(self, raw_src, raw_tgt, spec_src):
        raise NotImplementedError

    @staticmethod
    def freeze(model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
