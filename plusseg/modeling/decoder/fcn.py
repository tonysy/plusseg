import torch
import torch.nn.functional as F 

from torch import nn
try:
    from apex.parallel import SyncBatchNorm, DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


class FCN(nn.Module):
    def __init__(self,cfg, **kwargs):
        super(FCN, self).__init__()
        self._up_kwargs = kwargs
        self.aux_factor = cfg.MODEL.DECODER.AUX_FACTOR
        norm_layer=eval(cfg.MODEL.BATCH_NORM)

        self.fcn_head = FCNHead(
            cfg.MODEL.DECODER.FCN.IN_CHANNEL,
            cfg.MODEL.DECODER.FCN.CHANNEL_STRIDE,
            cfg.MODEL.DECODER.FCN.OUT_CHANNEL,
            cfg.MODEL.DECODER.FCN.DROPOUT,
            norm_layer=norm_layer)
        if self.aux_factor > 0:
            self.aux_fcn_head = FCNHead(
                cfg.MODEL.DECODER.FCN.AUX_IN_CHANNEL,
                cfg.MODEL.DECODER.FCN.CHANNEL_STRIDE,
                cfg.MODEL.DECODER.FCN.OUT_CHANNEL,
                cfg.MODEL.DECODER.FCN.DROPOUT,
                norm_layer=norm_layer
            )
    
    def forward(self, x, imsize):
        if len(x) == 2:
            c4, c5 = x[0], x[1]
        fcn_out = self.fcn_head(c5)
        fcn_out = F.interpolate(fcn_out, size=imsize, **self._up_kwargs)
        outputs = [fcn_out]
        if self.aux_factor > 0:
            aux_fcn_out = self.aux_fcn_head(c4)
            aux_fcn_out = F.interpolate(aux_fcn_out, imsize, **self._up_kwargs)
            outputs.append(fcn_out)
        return tuple(outputs)


class FCNHead(nn.Module):
    """
    Module use the fcn-8/16/32s for semantic segmentation
    """
    def __init__(self, in_channels, channel_stride, out_channels, dropout=0.1, norm_layer=nn.BatchNorm2d):
        """
        Arguments:
            in_channels_list (int): number of channels for each feature map that will be fed
            out_channels (int): number of channels of the FCN output.
        """
        super(FCNHead, self).__init__()

        inter_channels = in_channels // channel_stride
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.Dropout2d(dropout, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):

        return self.head_conv(x)


# _FCN_VARIANTS = Registry({
#     "FCN32s": FCN,
# })