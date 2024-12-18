import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
def std_loss(pred, target=None):
    pred_avg = pred.mean(1)[:, None]
    pred_std = torch.sqrt(((pred - pred_avg)**2).mean(1))
    return pred_std


@LOSSES.register_module
class SmoothL1_Std_Loss(nn.Module):
    def __init__(self,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 weight_L1=0.5,
                 weight_std=0.5):
        super(SmoothL1_Std_Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.weight_L1 = weight_L1
        self.weight_std = weight_std

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
        loss_bbox_L1 = self.loss_weight * self.weight_L1 * smooth_l1_loss(
            pred.sum(1),
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        loss_bbox_std = self.loss_weight * self.weight_std * std_loss(
            pred, None)
        loss_bbox = loss_bbox_L1 + loss_bbox_std
        print('loss_box_l1',loss_bbox_L1.data,'loss_bbox_std',loss_bbox_std.data)
        return loss_bbox
