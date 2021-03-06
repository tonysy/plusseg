from collections import OrderedDict

from torch import nn

from plusseg.modeling import registry
from plusseg.modeling.make_layers import conv_with_kaiming_uniform

from . import fpn as fpn_module
from . import resnet
from . import densenet

@registry.BACKBONES.register("R-50-C45")
@registry.BACKBONES.register("R-101-C45")
@registry.BACKBONES.register("R-152-C45")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.ENCODER.BACKBONE.RESNETS.BACKBONE_OUT_CHANNELS

    return model

@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channel_stage2 = cfg.MODEL.ENCODER.BACKBONE.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.ENCODER.BACKBONE.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channel_stage2,
            in_channel_stage2 * 2,
            in_channel_stage2 * 4,
            in_channel_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict[("body", body), ("fpn", fpn)])
    model.out_channels = out_channels

    return model


def build_backbone(cfg):
    assert cfg.MODEL.ENCODER.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.ENCODER.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.ENCODER.BACKBONE.CONV_BODY
        )
    # import pdb; pdb.set_trace()
    return registry.BACKBONES[cfg.MODEL.ENCODER.BACKBONE.CONV_BODY](cfg)