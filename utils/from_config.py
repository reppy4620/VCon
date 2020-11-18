from modules import *
from dataset import *


_model_dict = {
    'againvc': AgainVCModel,
    'autovc': AutoVCModel,
    'autovc-alpha': AutoVCAlphaModel,
    'fragmentvc': FragmentVCModel,
    'transformer': TransformerModel,
    'transformer-alpha': TransformerAlphaModel,
    'vqvc': VQVCModel,
}

_module_dict = {
    'againvc': AgainVCModule,
    'autovc': AutoVCModule,
    'autovc-alpha': AutoVCAlphaModule,
    'fragmentvc': FragmentVCModule,
    'transformer': TransformerModule,
    'transformer-alpha': TransformerAlphaModule,
    'vqvc': VQVCModule,
}

_data_module_dict = {
    'againvc': MelDataModule,
    'autovc': WavMelDataModule,
    'autovc-alpha': WavMelDataModule,
    'fragmentvc': Wav2VecMelDataModule,
    'transformer': MelDataModule,
    'transformer-alpha': MelDataModule,
    'vqvc': WavMelDataModule,
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)


def datamodule_from_config(params):
    return _data_module_dict[params.exp_name](params)
