import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss
import math


@weighted_loss
def hinge_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    difference = torch.abs(pred) - target
    loss = torch.max(torch.zeros(difference.shape).cuda(), difference).sum(1)
    print(pred*180/math.pi)
    return loss


@LOSSES.register_module
class HingeLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(HingeLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if pred.shape == target.shape:
            loss_bbox = self.loss_weight * hinge_loss(pred,
                                                      target,
                                                      weight,
                                                      beta=self.beta,
                                                      reduction=reduction,
                                                      avg_factor=avg_factor,
                                                      **kwargs)

        return loss_bbox
