import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def max_min_angle_loss(pred, target):
    min_angle = pred.min(dim=1)[0]
    max_angle = pred.max(dim=1)[0]
    loss = max_angle - min_angle
    return loss


@LOSSES.register_module
class MaxMinAngleLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MaxMinAngleLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * max_min_angle_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
