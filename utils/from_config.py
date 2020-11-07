from modules import *
from dataset import *


_model_dict = {
    'autovc': AutoVCModel,
    'vqvc': VQVCModel,
    'fragmentvc': FragmentVCModel,
    'transformer': TransformerModel
}

_module_dict = {
    'autovc': AutoVCModule,
    'vqvc': VQVCModule,
    'fragmentvc': FragmentVCModule,
    'transformer': TransformerModule
}

_data_module_dict = {
    'autovc': WavMelDataModule,
    'vqvc': WavMelDataModule,
    'fragmentvc': Wav2VecMelDataModule,
    'transformer': MelDataModule
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)


def datamodule_from_config(params):
    return _data_module_dict[params.exp_name](params)
