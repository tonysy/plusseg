from collections import OrderedDict

from torch import nn

from plusseg.modeling import registry
from plusseg.modeling.make_layers import conv_with_kaiming_uniform

from . import fcn

@registry.DECODERS.register('FCN')
def build_fcn_decoder(cfg):
    decoder = fcn.FCN(cfg)
    model = nn.Sequential(OrderedDict([('decoder', decoder)]))
    return model
    
def build_decoder(cfg):
    assert cfg.MODEL.DECODER.NAME in registry.DECODERS, \
        "cfg.MODEL.DECODER.NAME: {} are not registered in registry".format(
            cfg.MODEL.DECODER.NAME
        )
    # import pdb; pdb.set_trace()
    return registry.DECODERS[cfg.MODEL.DECODER.NAME](cfg)