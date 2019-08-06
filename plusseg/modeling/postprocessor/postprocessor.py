from collections import OrderedDict

from torch import nn

from plusseg.modeling import registry
from plusseg.modeling.make_layers import conv_with_kaiming_uniform


def build_post_processor(cfg):
    if cfg.MODEL.POSTPROCESSOR.NAME is '':
        return None
        
    assert cfg.MODEL.POSTPROCESSOR.NAME in registry.POSTPROCESSORS, \
        "cfg.MODEL.POSTPROCESSOR.NAME: {} are not registered in registry".format(
            cfg.MODEL.POSTPROCESSOR.NAME
        )
    # import pdb; pdb.set_trace()
    return registry.POSTPROCESSORS[cfg.MODEL.POSTPROCESSOR.NAME](cfg)