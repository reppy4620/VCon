import torch
# from fairseq.models.wav2vec import Wav2Vec2Model


# TODO
def load_pretrained_wav2vec(ckpt_path):
    """Load pretrained Wav2Vec model."""
    ckpt = torch.load(ckpt_path)
    # model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    # model.load_state_dict(ckpt["model"])
    # model.remove_pretraining_modules()
    # model.eval()
    # return model
