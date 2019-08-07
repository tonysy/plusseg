import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class SegmentationLossComputation(object):
    """
    Compute the loss for Semantic Segmentation
    
    """
    def __init__(self,
        aux_factor,
        ignore_index=-1,
        size_average=True,
        weight=None,):
        super(SegmentationLossComputation, self).__init__()
        self.aux_factor = aux_factor

        self.loss_criterion = CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)

    def __call__(self, inputs, target):
        if self.aux_factor > 0:
            pred, aux_pred = inputs[0], inputs[1]
            loss_pred = self.loss_criterion(pred, target)
            loss_aux_pred = self.loss_criterion(aux_pred, target)

            return loss_pred + self.aux_factor * loss_aux_pred
        else:
            pred = inputs[0]
            loss_pred = self.loss_criterion(pred, target)
            return loss_pred
