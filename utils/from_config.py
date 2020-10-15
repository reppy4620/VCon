from nn import *


_model_dict = {
    'autovc': NormalAutoVCModel,
    'autovc_normal': NormalAutoVCModel,
    'autovc_attn': AttnAutoVCModel,
    'autovc_attention': AttnAutoVCModel,
    'autovc_vq': VQAutoVCModel,
    'quartz': QuartzModel,
    'adain': AdaINVCModel,
    'adain_gan': AdaINVCModel,
    'vqvc': VQVCModel
}

_module_dict = {
    'autovc': AutoVCModule,
    'autovc_normal': AutoVCModule,
    'autovc_attn': AutoVCModule,
    'autovc_attention': AutoVCModule,
    'autovc_vq': VQAutoVCModule,
    'quartz': QuartzModule,
    'adain': AdaINVCModule,
    'adain_gan': AdaINGANModule,
    'vqvc': VQVCModule
}


def model_from_config(params):
    return _model_dict[params.exp_name](params)


def module_from_config(params):
    return _module_dict[params.exp_name](params)
