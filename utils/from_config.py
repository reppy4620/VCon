from nn import *


_model_dict = {
    'autovc': AutoVCModel,
    'autovc_vqvae': AutoVCBaseVQVAEModel,
    'quartz': QuartzModel,
}

_module_dict = {
    'autovc': AutoVCModule,
    'autovc_vqvae': AutoVCBaseVQVAEModule,
    'quartz': QuartzModule,
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)
