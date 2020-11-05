from modules import *


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
    'autovc': AutoVCDataModule,
    'vqvc': VQVCDataModule,
    'fragmentvc': FragmentVCDataModule,
    'transformer': TransformerDataModule
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)


def datamodule_from_config(params):
    return _data_module_dict[params.exp_name](params)
