from modules import *


_model_dict = {
    'autovc': AutoVCModel,
    'vqvc': VQVCModel,
    'tfm': TransformerModel
}

_module_dict = {
    'autovc': AutoVCModule,
    'vqvc': VQVCModule,
    'tfm': TransformerModule
}

_data_module_dict = {
    'autovc': AutoVCDataModule,
    'vqvc': VQVCDataModule,
    'tfm': TransformerDataModule
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)


def datamodule_from_config(params):
    return _data_module_dict[params.exp_name](params)
