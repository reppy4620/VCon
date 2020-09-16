from nn import *


_model_dict = {
    'autovc': AutoVCModel,
    'quartz': QuartzModel,
    'adain': AdaINVCModel,
    'adain_gan': AdaINVCModel,
}

_module_dict = {
    'autovc': AutoVCModule,
    'quartz': QuartzModule,
    'adain': AdaINVCModule,
    'adain_gan': AdaINGANModule,
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)
